#!/usr/bin/env python3
"""
Script to evaluate trained models.
"""

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.metrics import MetricsCalculator
from src.training.qlora_trainer import QLoRATrainer
from src.training.config import TrainingConfig
from datasets import load_from_disk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Base model name (for loading LoRA adapters)"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--input-column",
        type=str,
        default="input",
        help="Input column name"
    )
    parser.add_argument(
        "--reference-column",
        type=str,
        default="output",
        help="Reference output column name"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--skip-perplexity",
        action="store_true",
        help="Skip perplexity calculation (faster)"
    )
    parser.add_argument(
        "--benchmark-speed",
        action="store_true",
        help="Run inference speed benchmark"
    )
    
    args = parser.parse_args()
    
    # Load model
    logger.info("Loading model...")
    config = TrainingConfig(model_name=args.base_model)
    trainer = QLoRATrainer(config)
    trainer.load_trained_model(args.model_path)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_path}...")
    dataset = load_from_disk(args.dataset_path)
    eval_dataset = dataset[args.split]
    
    logger.info(f"Evaluating on {len(eval_dataset)} examples from {args.split} split")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model=trainer.model,
        tokenizer=trainer.tokenizer,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    metrics = evaluator.evaluate_dataset(
        dataset=eval_dataset,
        input_column=args.input_column,
        reference_column=args.reference_column,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        include_perplexity=not args.skip_perplexity
    )
    
    # Benchmark speed if requested
    if args.benchmark_speed:
        logger.info("\nRunning speed benchmark...")
        speed_metrics = evaluator.benchmark_inference_speed(
            dataset=eval_dataset,
            input_column=args.input_column,
            num_runs=3,
            max_new_tokens=args.max_new_tokens
        )
        metrics.update(speed_metrics)
    
    logger.info("\n" + "="*60)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
