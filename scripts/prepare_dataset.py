#!/usr/bin/env python3
"""
Script to prepare and validate datasets.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_builder import DomainDatasetBuilder
from src.data.preprocessing import DataPreprocessor
from src.data.validation import DataValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Prepare domain-specific dataset")
    
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=["healthcare", "legal", "finance"],
        help="Domain for the dataset"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="HuggingFace dataset name (optional)"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to input file (JSON or CSV)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Create sample dataset for testing"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples for sample dataset"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Training set proportion"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation set proportion"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Test set proportion"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip dataset validation"
    )
    
    args = parser.parse_args()
    
    # Initialize builder
    builder = DomainDatasetBuilder(domain=args.domain, output_dir=args.output_dir)
    
    # Load or create dataset
    if args.sample:
        logger.info(f"Creating sample dataset with {args.num_samples} samples")
        dataset = builder.create_sample_dataset(num_samples=args.num_samples)
    
    elif args.dataset_name:
        logger.info(f"Loading dataset from HuggingFace: {args.dataset_name}")
        dataset = builder.load_from_huggingface(args.dataset_name)
    
    elif args.input_file:
        logger.info(f"Loading dataset from file: {args.input_file}")
        if args.input_file.endswith('.json'):
            dataset = builder.load_from_json(args.input_file)
        elif args.input_file.endswith('.csv'):
            dataset = builder.load_from_csv(args.input_file)
        else:
            raise ValueError("Input file must be JSON or CSV")
    
    else:
        logger.info("Getting recommended datasets for domain")
        recommendations = builder.get_domain_specific_datasets()
        logger.info(f"Recommended datasets for {args.domain}:")
        for rec in recommendations:
            logger.info(f"  - {rec}")
        return
    
    # Preprocess
    logger.info("Preprocessing dataset...")
    preprocessor = DataPreprocessor(max_length=args.max_length)
    
    # Get text columns
    text_columns = [col for col in dataset.column_names if col in ["text", "input", "output", "instruction"]]
    
    if text_columns:
        dataset = preprocessor.apply_preprocessing(dataset, text_columns)
        dataset = preprocessor.remove_empty_examples(dataset, text_columns)
        dataset = preprocessor.filter_by_length(dataset)
        dataset = preprocessor.remove_duplicates(dataset)
    
    # Validate
    if not args.skip_validation:
        logger.info("Validating dataset...")
        validator = DataValidator(required_columns=text_columns)
        validation_results = validator.run_full_validation(dataset)
        
        # Print report
        report = validator.generate_validation_report(validation_results)
        print("\n" + report)
        
        if not validation_results["overall_valid"]:
            logger.warning("Dataset has validation issues!")
    
    # Split dataset
    logger.info("Splitting dataset...")
    dataset_dict = builder.split_dataset(
        dataset,
        train_size=args.train_split,
        val_size=args.val_split,
        test_size=args.test_split
    )
    
    logger.info(f"Dataset splits:")
    logger.info(f"  Train: {len(dataset_dict['train'])}")
    logger.info(f"  Validation: {len(dataset_dict['validation'])}")
    logger.info(f"  Test: {len(dataset_dict['test'])}")
    
    # Save
    output_name = f"{args.domain}_dataset"
    builder.save_dataset(dataset_dict, output_name)
    
    logger.info(f"✓ Dataset preparation complete!")
    logger.info(f"Saved to: {Path(args.output_dir) / output_name}")


if __name__ == "__main__":
    main()
