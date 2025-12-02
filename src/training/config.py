"""
QLoRA training configuration and hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class TrainingConfig:
    """Configuration for QLoRA training."""
    
    # Model settings
    model_name: str = "mistralai/Mistral-7B-v0.1"
    tokenizer_name: Optional[str] = None  # Uses model_name if None
    max_seq_length: int = 2048
    
    # LoRA settings
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Quantization settings
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True
    
    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    
    # Optimization
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "cosine"
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    output_dir: str = "models/checkpoints"
    save_total_limit: int = 3
    
    # Advanced settings
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    group_by_length: bool = True
    report_to: str = "none"  # Can be "wandb", "tensorboard", etc.
    
    # Dataset settings
    dataset_text_field: str = "text"
    packing: bool = False
    
    # Seed
    seed: int = 42
    
    def __post_init__(self):
        """Validate and process configuration."""
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name
        
        # Convert dtype string to torch dtype
        if self.bnb_4bit_compute_dtype == "bfloat16":
            self.compute_dtype = torch.bfloat16
        elif self.bnb_4bit_compute_dtype == "float16":
            self.compute_dtype = torch.float16
        else:
            self.compute_dtype = torch.float32
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


@dataclass
class FullFineTuneConfig(TrainingConfig):
    """Configuration for full fine-tuning (non-LoRA)."""
    
    # Override LoRA settings for full fine-tune
    use_lora: bool = False
    
    # Typically use smaller batch size for full fine-tune
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    
    # Often need lower learning rate
    learning_rate: float = 5e-5


@dataclass
class ComparisonConfig:
    """Configuration for comparing LoRA vs full fine-tune."""
    
    lora_config: TrainingConfig = field(default_factory=TrainingConfig)
    full_finetune_config: FullFineTuneConfig = field(default_factory=FullFineTuneConfig)
    
    # Comparison settings
    comparison_metrics: List[str] = field(
        default_factory=lambda: ["perplexity", "accuracy", "f1", "training_time", "memory_usage"]
    )
    save_comparison_report: bool = True
    comparison_output_dir: str = "results/comparison"
