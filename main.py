#!/usr/bin/env python
"""
AI Code Autocompletion - Main script

This script provides command-line access to various functionalities of the code autocompletion model:
1. Data preparation
2. Model training
3. Model evaluation
4. Interactive code completion

Example usage:
    # Prepare data
    python main.py prepare-data --dataset ieadoboe/python-function-examples --output data/processed/examples.json

    # Train model
    python main.py train --data data/processed/examples.json --output models/code-completion

    # Evaluate model
    python main.py evaluate --model models/code-completion --data data/processed/examples.json

    # Interactive completion
    python main.py complete "def process_image(image_path):\\n    # Load image"
"""

import argparse
import os
import sys
import json
from typing import List, Dict, Any

def prepare_data(args):
    """Prepare training data."""
    from src.data.load_data import load_code_dataset, extract_training_data, save_examples_to_disk
    
    # Load dataset
    dataset = load_code_dataset(
        dataset_name=args.dataset,
        split=args.split,
        max_examples=args.max_examples
    )
    
    # Extract examples
    examples = extract_training_data(dataset, max_examples=args.max_examples)
    
    # Save to disk
    save_examples_to_disk(examples, output_dir=os.path.dirname(args.output), filename=os.path.basename(args.output))
    
    print(f"Data preparation complete: {len(examples)} examples saved to {args.output}")

def train(args):
    """Train the model."""
    from src.data.load_data import load_examples_from_disk
    from src.features.tokenization import get_tokenizer, prepare_training_samples
    from src.models.train_model import train_code_completion_model
    
    # Load data
    examples = load_examples_from_disk(args.data)
    
    # Initialize tokenizer
    tokenizer = get_tokenizer(args.model_name)
    
    # Prepare training samples
    samples = prepare_training_samples(examples, tokenizer, max_length=args.max_length)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Train model
    model, tokenizer = train_code_completion_model(
        training_samples=samples,
        tokenizer=tokenizer,
        model_name=args.model_name,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        use_peft=not args.no_peft,
    )
    
    print(f"Training complete, model saved to {args.output}")

def evaluate(args):
    """Evaluate the model."""
    from src.data.load_data import load_examples_from_disk
    from src.features.tokenization import get_tokenizer
    from src.models.inference import load_model_for_inference
    from src.models.evaluate_model import (
        prepare_evaluation_samples, 
        evaluate_completions,
        plot_evaluation_metrics
    )
    
    # Load model
    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}, using base model: {args.base_model}")
        model_path = args.base_model
        use_peft = False
    else:
        model_path = args.model
        use_peft = not args.no_peft
    
    # Load tokenizer and model
    model, tokenizer = load_model_for_inference(
        model_path=model_path,
        use_peft=use_peft,
        base_model_name=args.base_model
    )
    
    # Load examples
    examples = load_examples_from_disk(args.data)
    
    # Prepare evaluation samples
    evaluation_samples = prepare_evaluation_samples(
        examples=examples, 
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        truncation_ratio=args.truncation_ratio
    )
    
    # Limit number of samples if specified
    if args.num_samples > 0 and args.num_samples < len(evaluation_samples):
        evaluation_samples = evaluation_samples[:args.num_samples]
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Evaluate completions
    metrics = evaluate_completions(
        evaluation_samples=evaluation_samples,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output
    )
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Number of samples: {metrics['num_samples']}")
    print(f"BLEU score: {metrics['bleu_score_mean']:.4f} ± {metrics['bleu_score_std']:.4f}")
    print(f"Exact match ratio: {metrics['exact_match_ratio_mean']:.4f} ± {metrics['exact_match_ratio_std']:.4f}")
    print(f"Edit distance: {metrics['edit_distance_mean']:.2f} ± {metrics['edit_distance_std']:.2f}")
    print(f"Normalized edit distance: {metrics['normalized_edit_distance_mean']:.4f} ± {metrics['normalized_edit_distance_std']:.4f}")
    
    # Plot metrics
    plot_evaluation_metrics(metrics, output_dir=args.output)
    
    print(f"Evaluation complete, results saved to {args.output}")

def complete(args):
    """Generate code completion."""
    from src.models.inference import generate_completion
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load model
    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}, using base model: {args.base_model}")
        model_path = args.base_model
        use_peft = False
        
        print(f"Loading base model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        from src.models.inference import load_model_for_inference
        model_path = args.model
        use_peft = not args.no_peft
        
        # Load tokenizer and model
        model, tokenizer = load_model_for_inference(
            model_path=model_path,
            use_peft=use_peft,
            base_model_name=args.base_model
        )
    
    # Generate completion
    completion = generate_completion(
        model=model,
        tokenizer=tokenizer,
        function_prefix=args.prefix,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.no_sample,
    )
    
    # Print result
    print("\nInput:")
    print(args.prefix)
    print("\nCompletion:")
    print(completion)

def interactive(args):
    """Interactive mode for code completion."""
    from src.models.inference import load_model_for_inference, generate_completion
    
    print("Loading model...")
    
    # Load model
    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}, using base model: {args.base_model}")
        model_path = args.base_model
        use_peft = False
    else:
        model_path = args.model
        use_peft = not args.no_peft
    
    # Load tokenizer and model
    model, tokenizer = load_model_for_inference(
        model_path=model_path,
        use_peft=use_peft,
        base_model_name=args.base_model
    )
    
    print("\nAI Code Autocompletion - Interactive Mode")
    print("Type 'exit' or 'quit' to end the session")
    print("Type 'settings' to view current generation settings")
    print("Type 'help' to view available commands")
    
    # Generation settings
    settings = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": not args.no_sample,
    }
    
    # Command handling
    def handle_settings_command(cmd_parts):
        if len(cmd_parts) == 1:
            print("\nCurrent settings:")
            for key, value in settings.items():
                print(f"  {key} = {value}")
            return
            
        if len(cmd_parts) != 3:
            print("Usage: settings <setting_name> <value>")
            return
            
        setting_name = cmd_parts[1]
        if setting_name not in settings:
            print(f"Unknown setting: {setting_name}")
            print(f"Available settings: {', '.join(settings.keys())}")
            return
            
        try:
            value = cmd_parts[2]
            if setting_name in ["max_tokens"]:
                settings[setting_name] = int(value)
            elif setting_name in ["temperature", "top_p"]:
                settings[setting_name] = float(value)
            elif setting_name in ["do_sample"]:
                settings[setting_name] = value.lower() in ["true", "yes", "1"]
            print(f"Updated {setting_name} = {settings[setting_name]}")
        except ValueError:
            print(f"Invalid value for {setting_name}: {value}")
    
    def handle_help_command():
        print("\nAvailable commands:")
        print("  exit, quit - Exit the program")
        print("  settings - View current generation settings")
        print("  settings <setting_name> <value> - Update a setting")
        print("  help - Show this help message")
    
    # Main interactive loop
    buffer = []
    while True:
        try:
            if not buffer:
                user_input = input("\n> ")
            else:
                user_input = input("... ")
                
            # Handle commands
            if not buffer and user_input.lower() in ["exit", "quit"]:
                break
            elif not buffer and user_input.lower() == "help":
                handle_help_command()
                continue
            elif not buffer and user_input.lower().startswith("settings"):
                handle_settings_command(user_input.split())
                continue
                
            # Handle code input
            if user_input.strip() == "" and buffer:
                # Empty line means we're done with input
                code_prefix = "\n".join(buffer)
                
                # Generate completion
                completion = generate_completion(
                    model=model,
                    tokenizer=tokenizer,
                    function_prefix=code_prefix,
                    max_new_tokens=settings["max_tokens"],
                    temperature=settings["temperature"],
                    top_p=settings["top_p"],
                    do_sample=settings["do_sample"],
                )
                
                print("\nCompletion:")
                print(completion)
                
                # Clear buffer for next input
                buffer = []
            else:
                # Add line to buffer
                buffer.append(user_input)
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            buffer = []
    
    print("Goodbye!")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AI Code Autocompletion")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Prepare data command
    prepare_parser = subparsers.add_parser("prepare-data", help="Prepare training data")
    prepare_parser.add_argument("--dataset", type=str, default="ieadoboe/python-function-examples", help="Dataset name")
    prepare_parser.add_argument("--split", type=str, default="train", help="Dataset split")
    prepare_parser.add_argument("--max-examples", type=int, default=1000, help="Maximum number of examples")
    prepare_parser.add_argument("--output", type=str, default="data/processed/training_examples.json", help="Output file")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data", type=str, default="data/processed/training_examples.json", help="Training data file")
    train_parser.add_argument("--model-name", type=str, default="Salesforce/codegen-350M-mono", help="Base model name")
    train_parser.add_argument("--output", type=str, default="models/code-completion", help="Output directory")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    train_parser.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps")
    train_parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    train_parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    train_parser.add_argument("--max-length", type=int, default=256, help="Maximum sequence length")
    train_parser.add_argument("--no-peft", action="store_true", help="Disable PEFT")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument("--model", type=str, default="models/code-completion", help="Model directory")
    eval_parser.add_argument("--base-model", type=str, default="Salesforce/codegen-350M-mono", help="Base model name")
    eval_parser.add_argument("--data", type=str, default="data/processed/training_examples.json", help="Evaluation data file")
    eval_parser.add_argument("--output", type=str, default="data/evaluation", help="Output directory")
    eval_parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens for evaluation")
    eval_parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum new tokens for generation")
    eval_parser.add_argument("--truncation-ratio", type=float, default=0.5, help="Ratio for truncating examples")
    eval_parser.add_argument("--num-samples", type=int, default=-1, help="Number of samples to evaluate")
    eval_parser.add_argument("--no-peft", action="store_true", help="Disable PEFT")
    
    # Complete command
    complete_parser = subparsers.add_parser("complete", help="Generate code completion")
    complete_parser.add_argument("prefix", type=str, help="Code prefix to complete")
    complete_parser.add_argument("--model", type=str, default="models/code-completion", help="Model directory")
    complete_parser.add_argument("--base-model", type=str, default="Salesforce/codegen-350M-mono", help="Base model name")
    complete_parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    complete_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    complete_parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling parameter")
    complete_parser.add_argument("--no-sample", action="store_true", help="Disable sampling")
    complete_parser.add_argument("--no-peft", action="store_true", help="Disable PEFT")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive code completion")
    interactive_parser.add_argument("--model", type=str, default="models/code-completion", help="Model directory")
    interactive_parser.add_argument("--base-model", type=str, default="Salesforce/codegen-350M-mono", help="Base model name")
    interactive_parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    interactive_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    interactive_parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling parameter")
    interactive_parser.add_argument("--no-sample", action="store_true", help="Disable sampling")
    interactive_parser.add_argument("--no-peft", action="store_true", help="Disable PEFT")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run command
    if args.command == "prepare-data":
        prepare_data(args)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "complete":
        complete(args)
    elif args.command == "interactive":
        interactive(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 