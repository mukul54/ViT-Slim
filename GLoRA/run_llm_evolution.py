#!/usr/bin/env python
# coding=utf-8

import os
import sys
import argparse
import logging
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from llm_evolution import LLMEvolutionSearcher
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def load_alpaca_dataset(dataset_name, subset_size=1000):
    """Load and subset a validation dataset for evolution search"""
    try:
        dataset = load_dataset(dataset_name)
        
        if "train" in dataset:
            data = dataset["train"]
        else:
            # Use the first split if "train" is not available
            first_key = list(dataset.keys())[0]
            data = dataset[first_key]
            
        # Take a subset for faster evaluation
        if subset_size and len(data) > subset_size:
            indices = list(range(len(data)))
            random.shuffle(indices)
            subset_indices = indices[:subset_size]
            data = data.select(subset_indices)
            
        return data
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        sys.exit(1)

class ValidationDataset(torch.utils.data.Dataset):
    """Simple validation dataset for GLoRA evolution search"""
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format examples - adapt this to your dataset format
        if "instruction" in item and "input" in item and "output" in item:
            if item["input"]:
                text = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
            else:
                text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
        else:
            # Fallback for different formats
            text = item.get("text", "")
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Prepare inputs for evaluation
        inputs = {
            "input_ids": encodings["input_ids"][0],
            "attention_mask": encodings["attention_mask"][0],
            "labels": encodings["input_ids"][0].clone(),
        }
        
        return inputs

def parse_args():
    parser = argparse.ArgumentParser(description="Run evolutionary search on a GLoRA model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained GLoRA model",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="tatsu-lab/alpaca",
        help="Dataset for evaluation during evolution",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=1000,
        help="Number of examples to use for evaluation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evolution_results",
        help="Directory to save evolution results",
    )
    parser.add_argument(
        "--evolution_epochs",
        type=int,
        default=20,
        help="Number of epochs for evolutionary search",
    )
    parser.add_argument(
        "--select_num",
        type=int,
        default=10,
        help="Number of top configurations to select",
    )
    parser.add_argument(
        "--population_num",
        type=int,
        default=50,
        help="Population size for evolutionary search",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="GLoRA rank for random configurations",
    )
    parser.add_argument(
        "--m_prob",
        type=float,
        default=0.2,
        help="Mutation probability",
    )
    parser.add_argument(
        "--crossover_num",
        type=int,
        default=25,
        help="Number of crossover candidates",
    )
    parser.add_argument(
        "--mutation_num",
        type=int,
        default=25,
        help="Number of mutation candidates",
    )
    parser.add_argument(
        "--param_limits",
        type=float,
        default=1.0,
        help="Maximum parameter budget in millions",
    )
    parser.add_argument(
        "--min_param_limits",
        type=float,
        default=0.0,
        help="Minimum parameter budget in millions",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda:0')",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set seeds for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", torch_dtype=torch.float16),
        args.model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load dataset
    logger.info(f"Loading dataset {args.dataset_name}")
    raw_data = load_alpaca_dataset(args.dataset_name, args.subset_size)
    val_dataset = ValidationDataset(raw_data, tokenizer)
    
    # Initialize evolution searcher
    logger.info("Initializing evolution search")
    searcher = LLMEvolutionSearcher(
        args=args,
        device=device,
        model=model,
        tokenizer=tokenizer,
        val_dataset=val_dataset,
        output_dir=args.output_dir
    )
    
    # Run evolution search
    logger.info("Starting evolution search")
    best_config = searcher.search()
    
    # Apply best config to model
    logger.info("Applying best configuration to model")
    i = 0
    for name, module in model.named_modules():
        if hasattr(module, 'eval_config'):
            if i < len(best_config):
                module.eval_config = best_config[i]
                i += 1
    
    # Save the best model
    best_model_dir = os.path.join(args.output_dir, "best_model")
    logger.info(f"Saving best model to {best_model_dir}")
    os.makedirs(best_model_dir, exist_ok=True)
    model.save_pretrained(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    
    logger.info("Evolution search complete!")

if __name__ == "__main__":
    main()