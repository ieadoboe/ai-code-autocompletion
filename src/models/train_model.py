"""
Training utilities for code completion models.
"""
import os
from typing import Dict, Tuple, Any, Optional
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    TrainerCallback
)

from src.features.tokenization import (
    get_tokenizer, 
    prepare_training_samples, 
    CodeCompletionDataset, 
    split_train_val
)
from src.models.peft_config import create_peft_config, create_peft_model

class SavePeftModelCallback(TrainerCallback):
    """
    Callback to save PEFT model during training.
    """
    def on_save(self, args, state, control, **kwargs):
        """Save the PEFT model adapter when checkpoint is saved."""
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs['model'].save_pretrained(checkpoint_dir)
        return control

def train_code_completion_model(
    training_samples,
    tokenizer=None,
    model_name: str = "Salesforce/codegen-350M-mono",
    output_dir: str = "./code-completion-model",
    num_epochs: int = 3,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    max_length: int = 256,
    use_peft: bool = True,
    peft_config = None,
) -> Tuple[Any, Any]:
    """
    Train a code completion model.
    
    Args:
        training_samples: Samples for training
        tokenizer: Tokenizer to use, if None will load from model_name
        model_name: Name of the model to use
        output_dir: Directory to save the model
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        grad_accum_steps: Gradient accumulation steps
        learning_rate: Learning rate
        weight_decay: Weight decay
        max_length: Maximum sequence length
        use_peft: Whether to use PEFT
        peft_config: PEFT configuration, if None a default configuration will be created
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer if not provided
    if tokenizer is None:
        print(f"Loading tokenizer: {model_name}")
        tokenizer = get_tokenizer(model_name)

    # Split data into train and validation
    train_samples, val_samples = split_train_val(training_samples)

    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")

    # Create datasets
    train_dataset = CodeCompletionDataset(train_samples, tokenizer, max_length)
    val_dataset = CodeCompletionDataset(val_samples, tokenizer, max_length)

    # Load and prepare model
    if use_peft:
        # Use PEFT with LoRA
        if peft_config is None:
            peft_config = create_peft_config()
        model = create_peft_model(model_name, peft_config)
    else:
        # Use standard fine-tuning
        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
        load_best_model_at_end=True,
    )

    # Create trainer
    callbacks = [SavePeftModelCallback()] if use_peft else []
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=callbacks,
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer

if __name__ == "__main__":
    # Example usage
    from src.data.load_data import load_examples_from_disk
    from src.features.tokenization import prepare_training_samples, get_tokenizer
    
    # Load data
    examples = load_examples_from_disk()
    tokenizer = get_tokenizer()
    samples = prepare_training_samples(examples, tokenizer)
    
    # Train model
    model, tokenizer = train_code_completion_model(samples, tokenizer) 