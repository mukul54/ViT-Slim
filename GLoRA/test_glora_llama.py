#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import GLoraConfig, get_peft_model

def parse_args():
    parser = argparse.ArgumentParser(description="Test GLoRA adapter with Llama 3.2-3B")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./glora_llama_test_output",
        help="The output directory where model checkpoints will be written",
    )
    parser.add_argument(
        "--r",
        type=int,
        default=8,
        help="LoRA attention dimension",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="My favorite animal is",
        help="Prompt to use for generation test",
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading base model: {args.model_path}")
    
    if args.use_4bit:
        # Quantization config for 4-bit loading
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            load_in_4bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        # Standard loading
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Define target modules for Llama 3.2
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Create GLoRA config
    peft_config = GLoraConfig(
        r=args.r,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    
    print(f"Applying GLoRA adapter with rank {args.r} to target modules: {target_modules}")
    
    # Apply GLoRA to model
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Test generation
    print(f"\nTesting generation with prompt: '{args.prompt}'")
    # Process input with padding settings
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token explicitly
    inputs = tokenizer(args.prompt, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],  # Add attention mask explicitly
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id  # Specify pad token ID
        )
    
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated output:\n{decoded_output}")
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving test model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    
    print("GLoRA test completed successfully.")

if __name__ == "__main__":
    main()