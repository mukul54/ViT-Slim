#!/usr/bin/env python
# coding=utf-8

import os
import sys
import argparse
import logging
import math
import random
import contextlib
import numpy as np
import torch
import copy
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from pathlib import Path
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False
from datasets import load_dataset
from peft import get_peft_model
from peft.tuners.glora import GLoraConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a LLM with GLoRA")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="tatsu-lab/alpaca",
        help="Dataset name or path for finetuning",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for training",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="Rank for GLoRA",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="Alpha parameter for GLoRA",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Dropout probability for GLoRA",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./glora_finetuned_model",
        help="Directory to save the finetuned model",
    )
    parser.add_argument(
        "--save_epochs",
        action="store_true",
        help="Whether to save model after each epoch",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5000,
        help="Save a checkpoint every X steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log metrics every X steps",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="glora-llama-finetuning",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,  # Slightly lower default learning rate for better stability
        help="Learning rate for training",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup over warmup_ratio fraction of total steps",
    )
    parser.add_argument(
        "--gradient_clip_val",
        type=float,
        default=1.0,
        help="Maximum gradient norm for gradient clipping",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,
        help="Weight decay to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use fp16 (mixed) precision instead of 32-bit",
    )
    # Distributed training arguments
    parser.add_argument(
        "--local_rank",
        "--local-rank",  # Support both formats (torch.distributed.launch uses hyphen)
        dest="local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training on GPUs"
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Whether to use distributed training"
    )
    parser.add_argument(
        "--ddp_backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo", "mpi"],
        help="DDP communication backend"
    )
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class AlpacaDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Set the padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format following the Alpaca dataset style
        if "instruction" in item and "input" in item and "output" in item:
            if item["input"]:
                prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n"
                target = item["output"]
            else:
                prompt = f"### Instruction:\n{item['instruction']}\n\n### Response:\n"
                target = item["output"]
        else:
            # Alternative format for other datasets
            prompt = item.get("prompt", "")
            target = item.get("completion", "")
        
        # Tokenize
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        
        # Construct input with response
        input_ids = prompt_ids + target_ids + [self.tokenizer.eos_token_id]
        
        # Construct labels: -100 for prompt (not included in loss), actual ids for target
        labels = [-100] * len(prompt_ids) + target_ids + [self.tokenizer.eos_token_id]
        
        # Truncate to max length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Pad to max_length
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            labels = labels + [-100] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_mask),
        }

def load_alpaca_dataset(dataset_name):
    """Load Alpaca dataset or similar instruction-following dataset"""
    try:
        # Try to load from HuggingFace datasets
        dataset = load_dataset(dataset_name)
        
        # Process based on the dataset structure
        if "train" in dataset:
            train_data = dataset["train"]
        else:
            # Use the first split if "train" is not available
            first_key = list(dataset.keys())[0]
            train_data = dataset[first_key]
            
        return train_data
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        sys.exit(1)

def get_glora_config(model, rank):
    """Create GLoRA configuration for the model"""
    # Define target modules for LLMs (attention layers)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Creating the GLoRA config - compatible with the new structure
    config = GLoraConfig(
        r=rank,
        target_modules=target_modules,
        task_type="CAUSAL_LM"
    )
    
    return config

def setup_distributed_training(args):
    """Initialize the distributed training environment"""
    # Check for environment variable (torchrun) first, then fallback to args
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    
    if args.local_rank == -1 or not args.distributed:
        # Not using distributed mode
        return False
    
    # Initialize the distributed process group
    init_process_group(backend=args.ddp_backend)
    
    # Set the device to the local rank
    torch.cuda.set_device(args.local_rank)
    
    logger.info(f"Running distributed training on rank {args.local_rank}")
    return True

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Setup distributed training if enabled
    is_distributed = setup_distributed_training(args)
    
    # Determine if this is the master process for logging
    is_master = args.local_rank == -1 or args.local_rank == 0
    
    # Set device for training    
    if is_distributed:
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer and model
    if is_master:
        logger.info(f"Loading base model: {args.model_name_or_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # When distributed training, we need to be careful with device placement
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if args.fp16 else torch.float32
    )
    # Move model to the correct device
    model = model.to(device)
    
    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    if is_master:
        logger.info(f"Loading dataset: {args.dataset_name}")
    raw_data = load_alpaca_dataset(args.dataset_name)
    
    # Create dataset and dataloader
    train_dataset = AlpacaDataset(raw_data, tokenizer, args.max_length)
    
    # Configure sampler for distributed training
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    
    # Use appropriate number of workers (system suggests 2)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=2 if is_distributed else 0,
        pin_memory=True
    )
    
    # Apply GLoRA adapter
    if is_master:
        logger.info(f"Applying GLoRA adapter with rank {args.rank}")
    peft_config = get_glora_config(model, args.rank)
    model = get_peft_model(model, peft_config)
    
    # CRITICAL: Ensure ALL model parameters are on the correct device after PEFT initialization
    # This is needed because some GLoRA parameters might still be on CPU
    model = model.to(device)
    
    # Wrap model with DDP if using distributed training
    if is_distributed:
        # Find unused parameters should be set to False for better performance,
        # but we need to test if this works with the GLoRA model
        # We'll set it to True for now which is safer but less efficient
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )
    
    # Print trainable parameters (master process only)
    if is_master:
        model.print_trainable_parameters() if not is_distributed else model.module.print_trainable_parameters()
    
    # Setup training
    # Use a more stable optimizer configuration with gradient clipping
    optimizer = torch.optim.AdamW(
        model.parameters() if not is_distributed else model.module.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    # Calculate total number of training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    # Create learning rate scheduler - using cosine with warmup for better stability
    lr_scheduler = get_scheduler(
        name="cosine",  # Cosine annealing is more stable than linear for fine-tuning
        optimizer=optimizer,
        num_warmup_steps=int(max_train_steps * args.warmup_ratio),
        num_training_steps=max_train_steps,
    )
    
    # Initialize W&B if enabled (master process only)
    if is_master and args.use_wandb:
        if not wandb_available:
            logger.warning("Weights & Biases not available. Please install it with `pip install wandb`")
        else:
            wandb_run_name = args.wandb_run_name
            if wandb_run_name is None:
                # Include info about distributed training in run name
                total_gpus = torch.cuda.device_count() if is_distributed else 1
                effective_batch = args.batch_size * total_gpus
                wandb_run_name = f"glora-llama-r{args.rank}-bs{effective_batch}-lr{args.learning_rate}-gpus{total_gpus}"
            
            wandb.init(
                project=args.wandb_project,
                name=wandb_run_name,
                config={
                    "model": args.model_name_or_path,
                    "dataset": args.dataset_name,
                    "rank": args.rank,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "epochs": args.num_train_epochs,
                    "warmup_ratio": args.warmup_ratio,
                    "weight_decay": args.weight_decay,
                    "max_length": args.max_length,
                    "fp16": args.fp16,
                }
            )
    
    # Training loop
    logger.info("Starting training...")
    model.train()
    total_loss = 0
    global_step = 0
    
    # Variables for tracking best model
    best_model_state = None
    best_loss = float('inf')
    eval_losses = {}
    
    for epoch in range(args.num_train_epochs):
        if is_master:
            logger.info(f"Starting epoch {epoch+1}/{args.num_train_epochs}")
        
        # Set the epoch for the distributed sampler
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        # CRITICAL FIX: Reset total loss counter at the start of each epoch
        total_loss = 0.0
        steps_in_epoch = 0
        
        # Only show progress bar on master process
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", disable=not is_master)
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass with mixed precision if fp16 is enabled (using newer API)
            with torch.amp.autocast('cuda') if args.fp16 else contextlib.nullcontext():
                outputs = model(**batch)
                loss = outputs.loss
            
            # Scale the loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            # Detect and handle NaN or inf gradients
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or batch_idx == len(train_dataloader) - 1:
                # Clip gradients to prevent exploding gradients (using user-specified value)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip_val)
                
                # Check for NaN or inf values in gradients
                valid_gradients = True
                for param in model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            valid_gradients = False
                            break
                
                # Skip update if gradients contain NaN or inf
                if not valid_gradients:
                    logger.warning(f"Skipping update at step {global_step} due to NaN or inf gradients")
                    optimizer.zero_grad()  # Clear bad gradients
                    continue
            # Multiply by accumulation steps to get the actual loss magnitude (for logging only)
            total_loss += loss.item() * args.gradient_accumulation_steps
            steps_in_epoch += 1
            
            # Update weights after accumulation steps
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or batch_idx == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Update progress bar
                # Calculate current average loss (properly by steps in epoch)
                current_avg_loss = total_loss / steps_in_epoch
                progress_bar.set_postfix({"loss": current_avg_loss})
                
                # Log metrics to W&B at regular intervals
                if global_step % args.logging_steps == 0 and args.use_wandb and wandb_available:
                    wandb.log({
                        "train/loss": current_avg_loss,
                        "train/learning_rate": lr_scheduler.get_last_lr()[0],
                        "train/step": global_step,
                        "train/epoch": epoch + (batch_idx / len(train_dataloader))
                    })
                
                # Save model checkpoint at regular intervals
                if global_step % args.save_steps == 0 and global_step > 0:
                    # Save current model state as latest checkpoint
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    # Save model and optimizer state
                    logger.info(f"Saving model checkpoint to {checkpoint_dir}")
                    # Make sure to access .module if using DDP
                    if is_distributed:
                        model.module.save_pretrained(checkpoint_dir)
                    else:
                        model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    
                    # Save optimizer and scheduler
                    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
                    torch.save(lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
                    
                    # Calculate validation loss on a small subset for model selection
                    model.eval()
                    eval_loss = 0.0
                    eval_steps = min(100, len(train_dataloader) // 10)  # Use 10% of data or 100 steps
                    with torch.no_grad():
                        for i, eval_batch in enumerate(train_dataloader):
                            if i >= eval_steps:
                                break
                            eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                            outputs = model(**eval_batch)
                            eval_loss += outputs.loss.item()
                    
                    eval_loss = eval_loss / eval_steps
                    eval_losses[global_step] = eval_loss
                    
                    # Log evaluation metrics
                    logger.info(f"Step {global_step}: Evaluation loss: {eval_loss:.4f}")
                    if args.use_wandb and wandb_available:
                        wandb.log({"eval/loss": eval_loss, "train/step": global_step})
                    
                    # Keep track of best model
                    if eval_loss < best_loss:
                        logger.info(f"New best model at step {global_step} with loss {eval_loss:.4f} (previous: {best_loss:.4f})")
                        best_loss = eval_loss
                        # Save as best model
                        best_model_dir = os.path.join(args.output_dir, "best_model")
                        os.makedirs(best_model_dir, exist_ok=True)
                        model.save_pretrained(best_model_dir)
                        tokenizer.save_pretrained(best_model_dir)
                    
                    # Return to training mode
                    model.train()
        
        # Log average loss for the epoch
        avg_loss = total_loss / steps_in_epoch
        logger.info(f"Epoch {epoch+1} - Average loss: {avg_loss:.4f}")
        
        # Save model after each epoch if requested (only master process)
        if is_master and args.save_epochs:
            epoch_dir = os.path.join(args.output_dir, f"epoch-{epoch+1}")
            logger.info(f"Saving model for epoch {epoch+1} to {epoch_dir}")
            os.makedirs(epoch_dir, exist_ok=True)
            
            # Make sure to access .module if using DDP
            if is_distributed:
                model.module.save_pretrained(epoch_dir)
            else:
                model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
        
        # Log epoch metrics to W&B
        if args.use_wandb and wandb_available:
            wandb.log({
                "train/epoch_loss": avg_loss,
                "train/epoch": epoch + 1
            })
    
    # Save the final model (only for master process)
    if is_master:
        final_model_dir = os.path.join(args.output_dir, "final_model")
        logger.info(f"Saving final model to {final_model_dir}")
        os.makedirs(final_model_dir, exist_ok=True)
        
        # Make sure to access .module if using DDP
        if is_distributed:
            model.module.save_pretrained(final_model_dir)
        else:
            model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
    
    # Log final training stats
    if args.use_wandb and wandb_available:
        # Create a plot of the evaluation loss over time
        if eval_losses:
            steps = list(eval_losses.keys())
            losses = list(eval_losses.values())
            wandb.log({"eval/loss_over_time": wandb.plot.line(
                table=[[x, y] for (x, y) in zip(steps, losses)],
                x="step",
                y="loss",
                title="Evaluation Loss Over Time")
            })
        wandb.finish()
    
    # Summarize training results
    logger.info("Training complete!")
    if best_loss < float('inf'):
        logger.info(f"Best model saved with evaluation loss: {best_loss:.4f}")
        # Create a simple results file
        with open(os.path.join(args.output_dir, "training_results.txt"), "w") as f:
            f.write(f"Model: {args.model_name_or_path}\n")
            f.write(f"Dataset: {args.dataset_name}\n")
            f.write(f"GLoRA rank: {args.rank}\n")
            f.write(f"Learning rate: {args.learning_rate}\n")
            f.write(f"Batch size: {args.batch_size}\n")
            f.write(f"Gradient accumulation steps: {args.gradient_accumulation_steps}\n")
            f.write(f"Number of epochs: {args.num_train_epochs}\n")
            f.write(f"Best evaluation loss: {best_loss:.4f}\n")

if __name__ == "__main__":
    # Support both torch.distributed.launch and torchrun
    # For torchrun, rank info is in environment variables
    main()
