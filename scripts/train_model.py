#!/usr/bin/env python3
"""
Script to train models using QLoRA.
"""

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.qlora_trainer import QLoRATrainer
from src.training.config import TrainingConfig
from datasets import load_from_disk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train LLM using QLoRA")
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Base model name"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to prepared dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/checkpoints",
        help="Output directory for model checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=64,
        help="LoRA r parameter"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--merge-adapters",
        action="store_true",
        help="Merge LoRA adapters with base model after training"
    )
    
    args = parser.parse_args()
    
    # Create training config
    config = TrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Initialize trainer
    logger.info("Initializing QLoRA trainer...")
    trainer = QLoRATrainer(config)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_path}...")
    dataset = load_from_disk(args.dataset_path)
    
    train_dataset = dataset['train']
    eval_dataset = dataset.get('validation', None)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Eval samples: {len(eval_dataset)}")
    
    # Train
    logger.info("Starting training...")
    metrics = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    logger.info("Training completed!")
    logger.info(f"Final metrics: {metrics}")
    
    # Merge adapters if requested
    if args.merge_adapters:
        logger.info("Merging LoRA adapters with base model...")
        merged_output = Path(args.output_dir) / "merged_model"
        trainer.merge_and_save(str(merged_output))
        logger.info(f"Merged model saved to {merged_output}")
    
    logger.info("✓ Training complete!")


if __name__ == "__main__":
    main()
