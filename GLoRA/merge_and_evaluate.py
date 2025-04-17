#!/usr/bin/env python
# coding=utf-8

import torch
import os
import argparse
import subprocess
from pathlib import Path
import tempfile
import shutil
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# def merge_adapter_with_base_model(adapter_path, output_path=None, device=None):
#     """
#     Merge a GLoRA adapter with its base model and save the result.
    
#     Args:
#         adapter_path: Path to the GLoRA adapter
#         output_path: Path to save the merged model (if None, will use a temp dir)
#         device: Device to load the model on (cuda:X or cpu)
        
#     Returns:
#         Path to the merged model
#     """
#     logger.info(f"Loading adapter from {adapter_path}")
    
#     # Load adapter config to get base model name
#     config = PeftConfig.from_pretrained(adapter_path)
#     base_model_id = config.base_model_name_or_path
#     logger.info(f"Base model: {base_model_id}")
    
#     # Determine whether to use a temporary directory
#     using_temp_dir = output_path is None
#     if using_temp_dir:
#         output_path = tempfile.mkdtemp()
#         logger.info(f"Created temporary output directory: {output_path}")
#     else:
#         os.makedirs(output_path, exist_ok=True)
#         logger.info(f"Will save merged model to: {output_path}")
    
#     # Load tokenizer
#     logger.info("Loading tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
#     # Set device map based on provided device
#     if device and ':' not in device:
#         device = f"cuda:{device}" if device.isdigit() else device
    
#     # Create proper device map - use specific device if provided, otherwise let it auto-distribute
#     device_map = {"": device} if device else "auto"
    
#     # Load base model
#     logger.info(f"Loading base model on device {device if device else 'auto'}...")
#     base_model = AutoModelForCausalLM.from_pretrained(
#         base_model_id, 
#         torch_dtype=torch.float16,
#         device_map=device_map
#     )
    
#     # Load adapter
#     logger.info("Loading GLoRA adapter...")
#     model = PeftModel.from_pretrained(base_model, adapter_path)
    
#     # Merge adapter weights with base model
#     logger.info("Merging adapter with base model...")
#     try:
#         merged_model = model.merge_and_unload()
#         logger.info("Successfully merged the model using merge_and_unload")
#     except (AttributeError, NotImplementedError) as e:
#         logger.warning(f"Could not use merge_and_unload: {e}")
#         logger.info("Using manual merging approach instead...")
        
#         # Manual merging approach
#         # Set eval_config for all layers if not already set
#         for module in model.modules():
#             if hasattr(module, 'eval_config') and module.eval_config is None and hasattr(module, 'configs'):
#                 # Use the first config as eval_config if not set
#                 if len(module.configs) > 0:
#                     module.eval_config = module.configs[0]
        
#         # Call merge on all Linear layers
#         for module in model.modules():
#             if hasattr(module, 'merge') and callable(module.merge) and not getattr(module, 'merged', False):
#                 module.merge()
        
#         # Use the model with merged weights
#         merged_model = model
#     logger.info("Merge complete!")
    
#     # Save the merged model
#     logger.info(f"Saving merged model to {output_path}")
#     merged_model.save_pretrained(output_path)
#     tokenizer.save_pretrained(output_path)
    
#     return output_path

def merge_adapter_with_base_model(adapter_path, output_path=None, device=None):
    """
    Merge a GLoRA adapter with its base model and save the result.
    
    Args:
        adapter_path: Path to the GLoRA adapter
        output_path: Path to save the merged model (if None, will use a temp dir)
        device: Device to load the model on (cuda:X or cpu)
        
    Returns:
        Path to the merged model
    """
    logger.info(f"Loading adapter from {adapter_path}")
    
    # Load adapter config to get base model name
    config = PeftConfig.from_pretrained(adapter_path)
    base_model_id = config.base_model_name_or_path
    logger.info(f"Base model: {base_model_id}")
    
    # Determine whether to use a temporary directory
    using_temp_dir = output_path is None
    if using_temp_dir:
        output_path = tempfile.mkdtemp()
        logger.info(f"Created temporary output directory: {output_path}")
    else:
        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Will save merged model to: {output_path}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Set device map based on provided device
    if device and ':' not in device:
        device = f"cuda:{device}" if device.isdigit() else device
    
    # Create proper device map - use specific device if provided, otherwise let it auto-distribute
    device_map = {"": device} if device else "auto"
    
    # Load base model
    logger.info(f"Loading base model on device {device if device else 'auto'}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        torch_dtype=torch.float16,
        device_map=device_map
    )
    
    # Load adapter
    logger.info("Loading GLoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Load best configuration if available
    parent_dir = os.path.dirname(adapter_path)
    config_paths = [
        os.path.join(parent_dir, "best_evolution_config.json"),
        os.path.join(parent_dir, "best_glora_config.json")
    ]
    
    config_loaded = False
    for config_path in config_paths:
        if os.path.exists(config_path):
            logger.info(f"Loading best configuration from {config_path}")
            import json
            with open(config_path, 'r') as f:
                best_config = json.load(f)
                
            # Apply configurations
            config_count = 0
            for name, module in model.named_modules():
                if hasattr(module, 'eval_config'):
                    if config_count < len(best_config):
                        module.eval_config = best_config[config_count]
                        config_count += 1
            
            logger.info(f"Applied configurations to {config_count} modules")
            config_loaded = True
            break
            
    if not config_loaded:
        logger.warning("No best configuration found. Using default configurations.")
    
    # Merge adapter weights with base model
    logger.info("Merging adapter with base model...")
    merged_model = model.merge_and_unload()
    logger.info("Successfully merged the model using merge_and_unload")
    logger.info("Merge complete!")
    
    # Save the merged model
    logger.info(f"Saving merged model to {output_path}")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    return output_path

def run_lm_eval(model_path, tasks, output_file, batch_size, device=None, args=None):
    """
    Run LM-Eval-Harness on the merged model with appropriate few-shot settings.
    Runs each benchmark separately with its own few-shot setting.
    
    Args:
        model_path: Path to the model to evaluate
        tasks: Comma-separated list of tasks to evaluate on
        output_file: Path to save the evaluation results
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        args: Command line arguments
    
    Returns:
        True if all evaluations succeed, False otherwise
    """
    # Parse tasks into a list
    task_list = [t.strip() for t in tasks.split(',')]
    logger.info(f"Running evaluation on tasks: {task_list}")
    
    # Format the device argument
    if device and ':' not in device:
        device = f"cuda:{device}" if device.isdigit() else device
    
    # Mapping of tasks to their few-shot settings
    task_shots = {}
    
    # Configure few-shot settings for each benchmark type
    for task in task_list:
        if 'arc' in task.lower():
            task_shots[task] = args.arc_fewshot
        elif 'hellaswag' in task.lower():
            task_shots[task] = args.hellaswag_fewshot
        elif 'mmlu' in task.lower():
            task_shots[task] = args.mmlu_fewshot
        elif 'truthfulqa' in task.lower():
            task_shots[task] = args.truthfulqa_fewshot
        else:
            # Default to 0-shot for unknown tasks
            task_shots[task] = 0
    
    # Results storage
    all_results = {}
    overall_success = True
    
    # Run evaluation for each task with its specific few-shot setting
    output_file_base, output_file_ext = os.path.splitext(output_file)
    for task, shots in task_shots.items():
        task_output_file = f"{output_file_base}_{task}{output_file_ext}"
        logger.info(f"Evaluating {task} with {shots}-shot examples")
            
        # Build the command for this specific task
        cmd = [
            "lm-eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", task,
            "--num_fewshot", str(shots),
            "--batch_size", f"{batch_size}"
        ]
        
        # Add device if specified
        if device:
            cmd.extend(["--device", device])
        
        # Add output path
        cmd.extend(["--output_path", task_output_file])
        
        # Run the command
        logger.info(f"Running command: {' '.join(cmd)}")
        try:
            process = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Evaluation of {task} complete! Results saved to {task_output_file}")
            
            # Extract and display results summary
            if "Results:" in process.stdout:
                result_summary = process.stdout.split("Results:", 1)[-1].strip()
                logger.info(f"Summary for {task}:")
                logger.info(result_summary)
                all_results[task] = result_summary
        except subprocess.CalledProcessError as e:
            logger.error(f"Evaluation of {task} failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            overall_success = False
    
    # Create a combined results file if needed
    if len(all_results) > 0:
        logger.info(f"Finished evaluating all tasks. Results saved to individual files.")
    
    return overall_success

def parse_args():
    parser = argparse.ArgumentParser(description="Merge GLoRA adapter with base model and run evaluations")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="./best_model",
        help="Path to the GLoRA adapter directory"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the merged model (if None, will use a temporary directory)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="arc_challenge,hellaswag,mmlu,truthfulqa_mc1",
        help="Comma-separated list of tasks to evaluate on"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--arc_fewshot",
        type=int,
        default=25,
        help="Number of few-shot examples for ARC benchmark (default: 25)"
    )
    parser.add_argument(
        "--hellaswag_fewshot",
        type=int,
        default=10,
        help="Number of few-shot examples for HellaSwag benchmark (default: 10)"
    )
    parser.add_argument(
        "--mmlu_fewshot",
        type=int,
        default=5,
        help="Number of few-shot examples for MMLU benchmark (default: 5)"
    )
    parser.add_argument(
        "--truthfulqa_fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples for TruthfulQA benchmark (default: 0)"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="glora_eval_results.json",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g., 'cuda:0', '0', or 'cpu')"
    )
    parser.add_argument(
        "--keep_merged_model",
        action="store_true",
        help="Keep the merged model after evaluation (only relevant when output_path is None)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Merge adapter with base model
    merged_model_path = merge_adapter_with_base_model(
        args.adapter_path,
        args.output_path,
        args.device
    )
    
    try:
        # Run evaluation
        success = run_lm_eval(
            merged_model_path,
            args.tasks,
            args.results_file,
            args.batch_size,
            args.device,
            args
        )
        
        if success:
            logger.info(f"Evaluation completed successfully!")
        else:
            logger.warning("Evaluation did not complete successfully.")
    finally:
        # Clean up temporary directory if used and not keeping
        if args.output_path is None and not args.keep_merged_model:
            logger.info(f"Cleaning up temporary directory: {merged_model_path}")
            shutil.rmtree(merged_model_path)
        elif args.output_path is None:
            logger.info(f"Keeping merged model at: {merged_model_path}")

if __name__ == "__main__":
    main()
