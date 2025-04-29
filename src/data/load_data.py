"""
Data loading utilities for the Code Autocompletion project.
"""
import os
from typing import Dict, List, Tuple, Any, Optional
from datasets import load_dataset

def load_code_dataset(dataset_name: str = "ieadoboe/python-function-examples", 
                      split: str = "train", 
                      max_examples: int = None) -> Any:
    """
    Load datasets from the Code_Search_Net repository.
    
    Args:
        dataset_name: Name of the dataset to load from HuggingFace
        split: Dataset split to load (train, validation, test)
        max_examples: Maximum number of examples to load
        
    Returns:
        Dataset object containing the code examples
    """
    # Load the dataset from HuggingFace
    dataset = load_dataset(dataset_name, split=split)
    
    # If max_examples is specified, truncate the dataset
    if max_examples is not None and max_examples < len(dataset):
        dataset = dataset.select(range(max_examples))
    
    print(f"Loaded {len(dataset)} examples from {dataset_name} ({split})")
    return dataset

def extract_training_data(dataset, max_examples: int = 1000) -> List[Dict]:
    """
    Extract training examples from the dataset.
    
    Args:
        dataset: The HuggingFace dataset containing code functions
        max_examples: Maximum number of examples to extract
        
    Returns:
        List of dictionaries containing function data
    """
    examples = []
    for example in dataset:
        code = example.get("code", "")
        if len(code.strip()) < 20:  # Skip very short functions
            continue
            
        examples.append({
            "function_name": example.get("func_name", ""),
            "docstring": example.get("docstring", ""),
            "code": code,
            "language": example.get("language", "python"),
        })

        if len(examples) >= max_examples:
            break

    print(f"Extracted {len(examples)} examples")
    return examples

def save_examples_to_disk(examples: List[Dict], 
                          output_dir: str = "data/processed", 
                          filename: str = "training_examples.json"):
    """
    Save extracted examples to disk in JSON format.
    
    Args:
        examples: List of example dictionaries to save
        output_dir: Directory to save the examples
        filename: Name of the output file
    """
    import json
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save examples to file
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)
    
    print(f"Saved {len(examples)} examples to {output_path}")
    
def load_examples_from_disk(input_path: str = "data/processed/training_examples.json") -> List[Dict]:
    """
    Load training examples from disk.
    
    Args:
        input_path: Path to the JSON file containing training examples
        
    Returns:
        List of example dictionaries
    """
    import json
    
    with open(input_path, 'r') as f:
        examples = json.load(f)
    
    print(f"Loaded {len(examples)} examples from {input_path}")
    return examples

if __name__ == "__main__":
    # Example usage
    dataset = load_code_dataset(max_examples=1000)
    examples = extract_training_data(dataset, max_examples=1000)
    save_examples_to_disk(examples) 