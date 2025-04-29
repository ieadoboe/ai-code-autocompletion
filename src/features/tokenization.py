"""
Tokenization and feature preparation utilities for code completion.
"""
import random
from typing import Dict, List, Any, Tuple
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

def get_tokenizer(model_name: str = "Salesforce/codegen-350M-mono"):
    """
    Load the tokenizer for the specified model.
    
    Args:
        model_name: Name of the model to load the tokenizer for
        
    Returns:
        Tokenizer for the specified model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure padding token is set
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"Setting pad_token_id to eos_token_id: {tokenizer.pad_token_id}")
        else:
            tokenizer.pad_token_id = 0
            print("Setting pad_token_id to 0")
    
    return tokenizer

def prepare_training_samples(examples: List[Dict], 
                            tokenizer: Any, 
                            max_length: int = 256) -> List[Dict]:
    """
    Prepare training samples from code examples.
    
    Args:
        examples: List of code examples
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        List of training samples with input_ids and labels
    """
    training_samples = []

    for example in examples:
        code = example["code"]
        # Skip very short functions
        if len(code.strip()) < 20:
            continue

        # Tokenize the code
        tokenized = tokenizer(code, truncation=True, max_length=max_length)
        input_ids = tokenized["input_ids"]

        # Create training examples
        seq_length = len(input_ids)
        if seq_length > 20:
            for _ in range(3):
                # Decide how much to keep (50-90% of tokens)
                keep_percent = random.uniform(0.5, 0.9)
                keep_tokens = int(seq_length * keep_percent)

                # Create input/target pairs
                input_sample = input_ids[:keep_tokens]
                target_sample = input_ids[keep_tokens:]

                training_samples.append({
                    "input_ids": input_sample,
                    "labels": target_sample,
                })

    print(f"Created {len(training_samples)} training samples from {len(examples)} examples")
    return training_samples

class CodeCompletionDataset(Dataset):
    """
    Dataset for code completion model training.
    """
    def __init__(self, samples: List[Dict], tokenizer: Any, max_length: int = 256):
        """
        Initialize the dataset.
        
        Args:
            samples: List of training samples
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Get input and target
        input_ids = sample["input_ids"]
        labels = sample["labels"]

        # Combine input with labels for training
        combined_ids = input_ids + labels

        # Handle truncation if needed
        if len(combined_ids) > self.max_length:
            combined_ids = combined_ids[: self.max_length]

        # Create attention mask
        attention_mask = [1] * len(combined_ids)

        # Pad sequences if needed
        padding_length = self.max_length - len(combined_ids)
        if padding_length > 0:
            combined_ids = combined_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        # Set up labels (set to -100 for input portion to ignore in loss)
        labels = [-100] * len(input_ids) + combined_ids[len(input_ids) :]

        # Ensure all sequences have the right length
        if len(labels) > self.max_length:
            labels = labels[: self.max_length]
        elif len(labels) < self.max_length:
            labels = labels + [-100] * (self.max_length - len(labels))

        return {
            "input_ids": torch.tensor(combined_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }


def split_train_val(samples: List[Dict], val_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
    """
    Split samples into training and validation sets.
    
    Args:
        samples: List of samples to split
        val_ratio: Ratio of validation samples
        
    Returns:
        Tuple of (training_samples, validation_samples)
    """
    val_size = int(len(samples) * val_ratio)
    train_samples = samples[:-val_size]
    val_samples = samples[-val_size:]
    
    print(f"Split {len(samples)} samples into {len(train_samples)} training and {len(val_samples)} validation samples")
    return train_samples, val_samples

if __name__ == "__main__":
    # Example usage
    from src.data.load_data import load_examples_from_disk
    
    examples = load_examples_from_disk()
    tokenizer = get_tokenizer()
    samples = prepare_training_samples(examples, tokenizer)
    train_samples, val_samples = split_train_val(samples)
    
    # Create datasets
    train_dataset = CodeCompletionDataset(train_samples, tokenizer)
    val_dataset = CodeCompletionDataset(val_samples, tokenizer)
    
    print(f"Created train dataset with {len(train_dataset)} samples")
    print(f"Created validation dataset with {len(val_dataset)} samples") 