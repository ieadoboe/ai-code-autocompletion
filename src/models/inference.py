"""
Inference utilities for code completion models.
"""
import torch
from typing import Dict, Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model_for_inference(
    model_path: str, 
    use_peft: bool = True, 
    base_model_name: str = "Salesforce/codegen-350M-mono",
    device: str = None
) -> tuple:
    """
    Load a model for inference.
    
    Args:
        model_path: Path to the model
        use_peft: Whether the model uses PEFT
        base_model_name: Name of the base model, required for PEFT
        device: Device to use for inference
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path} to {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Ensure padding token is set
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"Setting pad_token_id to eos_token_id: {tokenizer.pad_token_id}")
        else:
            tokenizer.pad_token_id = 0
            print("Setting pad_token_id to 0")
    
    # Load model
    if use_peft:
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        # Load PEFT adapter
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        # Load model directly
        model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Move model to device
    model = model.to(device)
    
    return model, tokenizer

def generate_completion(
    model: Any, 
    tokenizer: Any, 
    function_prefix: str, 
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.95,
    do_sample: bool = True,
    num_return_sequences: int = 1,
) -> str:
    """
    Generate a completion for a function prefix.
    
    Args:
        model: Model to use for generation
        tokenizer: Tokenizer to use for generation
        function_prefix: Function prefix to complete
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Temperature for sampling
        top_p: Top-p sampling parameter
        do_sample: Whether to use sampling
        num_return_sequences: Number of sequences to generate
        
    Returns:
        Generated completion
    """
    # Get device from model
    device = next(model.parameters()).device
    
    # Tokenize and move to appropriate device
    inputs = tokenizer(function_prefix, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate completion
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode the generated tokens
    completed_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Return only the newly generated part
    completion = completed_code[len(function_prefix):]
    
    return completion

def batch_generate_completions(
    model: Any, 
    tokenizer: Any, 
    function_prefixes: List[str],
    **kwargs
) -> List[str]:
    """
    Generate completions for multiple function prefixes.
    
    Args:
        model: Model to use for generation
        tokenizer: Tokenizer to use for generation
        function_prefixes: Function prefixes to complete
        **kwargs: Additional arguments to pass to generate_completion
        
    Returns:
        List of generated completions
    """
    completions = []
    
    for prefix in function_prefixes:
        completion = generate_completion(model, tokenizer, prefix, **kwargs)
        completions.append(completion)
    
    return completions

if __name__ == "__main__":
    # Example usage
    import os
    
    # Check if model exists
    model_path = "./code-completion-model"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}, using base model")
        model_path = "Salesforce/codegen-350M-mono"
        use_peft = False
    else:
        use_peft = True
    
    # Load model and tokenizer
    model, tokenizer = load_model_for_inference(
        model_path=model_path,
        use_peft=use_peft,
    )
    
    # Test with some examples
    test_prefixes = [
        "def train_model(X_train, y_train):\n    # Create TensorFlow model\n    model = tf.keras",
        "def process_image(image_path):\n    # Load and preprocess image\n    import numpy as np\n    img = ",
        "def create_bert_classifier():\n    # Initialize a BERT model from HuggingFace\n    from transformers import ",
    ]
    
    for prefix in test_prefixes:
        completion = generate_completion(model, tokenizer, prefix)
        print(f"\nPrefix:\n{prefix}")
        print(f"\nCompletion:\n{completion}")
        print("-" * 50) 