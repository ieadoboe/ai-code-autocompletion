"""
Parameter-Efficient Fine-Tuning (PEFT) configuration for code completion models.
"""
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM
import torch

def create_peft_config(
    r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    task_type: TaskType = TaskType.CAUSAL_LM,
):
    """
    Create a LoRA configuration for PEFT.
    
    Args:
        r: Rank of the LoRA update matrices
        lora_alpha: LoRA alpha parameter
        lora_dropout: Dropout probability for LoRA layers
        task_type: Task type for the model
        
    Returns:
        LoRA configuration
    """
    return LoraConfig(
        task_type=task_type,
        inference_mode=False,  # for fine-tuning
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

def create_peft_model(model_name: str = "Salesforce/codegen-350M-mono", peft_config=None):
    """
    Create a PEFT model with the given configuration.
    
    Args:
        model_name: Name of the base model
        peft_config: PEFT configuration, if None a default configuration will be created
        
    Returns:
        PEFT model with the LoRA adapter
    """
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create default PEFT config if not provided
    if peft_config is None:
        peft_config = create_peft_config()
    
    # Add LoRA adapter to model
    peft_model = get_peft_model(model, peft_config)
    
    # Print parameter stats
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
    
    return peft_model

def move_model_to_device(model, device=None):
    """
    Move model to the specified device.
    
    Args:
        model: Model to move
        device: Device to move the model to, if None will use CUDA if available
        
    Returns:
        Model on the specified device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Moving model to device: {device}")
    return model.to(device)

if __name__ == "__main__":
    # Example usage
    peft_config = create_peft_config()
    model = create_peft_model(peft_config=peft_config)
    model = move_model_to_device(model) 