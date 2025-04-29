"""
Helper functions for the code autocompletion project.
"""
import os
import time
import logging
from typing import Dict, List, Any, Optional, Callable

def setup_logging(log_file: str = None, level: int = logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file, if None will log to console
        level: Logging level
    """
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Add file handler if log file is provided
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger

def timeit(func: Callable) -> Callable:
    """
    Decorator to measure execution time of a function.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def ensure_directory_exists(directory: str):
    """
    Ensure a directory exists, create it if it doesn't.
    
    Args:
        directory: Directory path
    """
    os.makedirs(directory, exist_ok=True)
    
def count_parameters(model) -> Dict[str, int]:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params,
        "trainable_percent": 100 * trainable_params / total_params if total_params > 0 else 0,
    }

def pretty_print_dict(d: Dict, indent: int = 0, key_width: int = 20):
    """
    Pretty print a dictionary.
    
    Args:
        d: Dictionary to print
        indent: Indentation level
        key_width: Width for keys
    """
    for key, value in d.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            pretty_print_dict(value, indent + 2)
        else:
            print(" " * indent + f"{key:{key_width}}: {value}")

def find_optimal_batch_size(
    model, 
    tokenizer, 
    start_size: int = 1, 
    max_size: int = 64, 
    sequence_length: int = 512, 
    step: int = 1
) -> int:
    """
    Find the optimal batch size for a model given GPU memory.
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer
        start_size: Starting batch size
        max_size: Maximum batch size to try
        sequence_length: Sequence length
        step: Step size for batch size
        
    Returns:
        Optimal batch size
    """
    import torch
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Running on CPU, batch size optimization not applicable")
        return start_size
        
    model = model.to(device)
    
    # Try increasing batch sizes
    batch_size = start_size
    while batch_size <= max_size:
        try:
            # Create dummy batch
            dummy_input_ids = torch.randint(
                0, tokenizer.vocab_size, (batch_size, sequence_length), 
                device=device
            )
            dummy_attention_mask = torch.ones(
                (batch_size, sequence_length), 
                device=device
            )
            
            # Forward pass
            with torch.no_grad():
                outputs = model(
                    input_ids=dummy_input_ids,
                    attention_mask=dummy_attention_mask
                )
            
            # Clear memory
            del outputs, dummy_input_ids, dummy_attention_mask
            torch.cuda.empty_cache()
            
            # Increase batch size
            print(f"Batch size {batch_size} successful")
            batch_size += step
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Out of memory, use previous batch size
                print(f"Out of memory with batch size {batch_size}")
                return max(batch_size - step, start_size)
            else:
                # Other error
                raise e
    
    return batch_size - step 