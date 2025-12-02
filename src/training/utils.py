"""
Utility functions for training.
"""

import torch
import logging
from typing import Dict, Any
import psutil
import GPUtil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_gpu_utilization():
    """Print current GPU memory utilization."""
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            logger.info(
                f"GPU {gpu.id}: {gpu.name} - "
                f"Memory Used: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB "
                f"({gpu.memoryUtil*100:.1f}%)"
            )
    except Exception as e:
        logger.warning(f"Could not get GPU info: {e}")


def print_memory_usage():
    """Print CPU and GPU memory usage."""
    # CPU memory
    cpu_memory = psutil.virtual_memory()
    logger.info(f"CPU Memory: {cpu_memory.used / 1e9:.2f}GB / {cpu_memory.total / 1e9:.2f}GB ({cpu_memory.percent}%)")
    
    # GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            logger.info(f"GPU {i} Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")


def get_model_size(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Calculate model size and parameter counts.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with size information
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percentage": 100 * trainable_params / total_params,
        "size_mb": size_all_mb,
        "size_gb": size_all_mb / 1024
    }


def log_model_info(model: torch.nn.Module):
    """
    Log detailed model information.
    
    Args:
        model: PyTorch model
    """
    info = get_model_size(model)
    
    logger.info("=" * 60)
    logger.info("MODEL INFORMATION")
    logger.info("=" * 60)
    logger.info(f"Total Parameters: {info['total_params']:,}")
    logger.info(f"Trainable Parameters: {info['trainable_params']:,} ({info['trainable_percentage']:.2f}%)")
    logger.info(f"Model Size: {info['size_mb']:.2f} MB ({info['size_gb']:.2f} GB)")
    logger.info("=" * 60)


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def count_parameters(model: torch.nn.Module) -> tuple:
    """
    Count trainable and total parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def freeze_layers(model: torch.nn.Module, num_layers_to_freeze: int):
    """
    Freeze the first N layers of the model.
    
    Args:
        model: PyTorch model
        num_layers_to_freeze: Number of layers to freeze
    """
    layers = list(model.named_parameters())
    
    for i, (name, param) in enumerate(layers):
        if i < num_layers_to_freeze:
            param.requires_grad = False
            logger.info(f"Froze layer: {name}")
    
    trainable, total = count_parameters(model)
    logger.info(f"After freezing: {trainable:,} / {total:,} parameters trainable")


def setup_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")
