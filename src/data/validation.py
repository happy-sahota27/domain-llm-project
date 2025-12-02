"""
Dataset validation utilities to ensure data quality.
"""

import logging
from typing import Dict, List, Optional, Set
from datasets import Dataset
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validate dataset quality and integrity."""
    
    def __init__(self, required_columns: Optional[List[str]] = None):
        """
        Initialize validator.
        
        Args:
            required_columns: List of required column names
        """
        self.required_columns = required_columns or []
        logger.info("Initialized DataValidator")
    
    def validate_schema(self, dataset: Dataset) -> bool:
        """
        Validate that dataset has required columns.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            True if valid, False otherwise
        """
        dataset_columns = set(dataset.column_names)
        required_set = set(self.required_columns)
        
        missing_columns = required_set - dataset_columns
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        logger.info("Schema validation passed")
        return True
    
    def check_null_values(self, dataset: Dataset, columns: Optional[List[str]] = None) -> Dict:
        """
        Check for null/empty values in dataset.
        
        Args:
            dataset: Dataset to check
            columns: Specific columns to check (checks all if None)
            
        Returns:
            Dictionary with null value counts per column
        """
        if columns is None:
            columns = dataset.column_names
        
        null_counts = {}
        
        for col in columns:
            null_count = 0
            for example in dataset:
                value = example.get(col)
                if value is None or (isinstance(value, str) and not value.strip()):
                    null_count += 1
            
            null_counts[col] = {
                "count": null_count,
                "percentage": (null_count / len(dataset)) * 100
            }
        
        # Log columns with null values
        for col, info in null_counts.items():
            if info["count"] > 0:
                logger.warning(
                    f"Column '{col}' has {info['count']} null values ({info['percentage']:.2f}%)"
                )
        
        return null_counts
    
    def check_duplicates(self, dataset: Dataset, column: str = "text") -> Dict:
        """
        Check for duplicate entries.
        
        Args:
            dataset: Dataset to check
            column: Column to check for duplicates
            
        Returns:
            Dictionary with duplicate statistics
        """
        seen = set()
        duplicates = []
        
        for idx, example in enumerate(dataset):
            value = example.get(column, "")
            if value in seen:
                duplicates.append(idx)
            else:
                seen.add(value)
        
        duplicate_stats = {
            "num_duplicates": len(duplicates),
            "percentage": (len(duplicates) / len(dataset)) * 100,
            "duplicate_indices": duplicates[:100]  # First 100 for reference
        }
        
        if duplicate_stats["num_duplicates"] > 0:
            logger.warning(
                f"Found {duplicate_stats['num_duplicates']} duplicates "
                f"({duplicate_stats['percentage']:.2f}%)"
            )
        else:
            logger.info("No duplicates found")
        
        return duplicate_stats
    
    def validate_text_length(
        self,
        dataset: Dataset,
        column: str = "text",
        min_length: int = 10,
        max_length: int = 2048
    ) -> Dict:
        """
        Validate text lengths are within acceptable range.
        
        Args:
            dataset: Dataset to validate
            column: Column to check
            min_length: Minimum acceptable length (words)
            max_length: Maximum acceptable length (words)
            
        Returns:
            Dictionary with length validation statistics
        """
        too_short = 0
        too_long = 0
        lengths = []
        
        for example in dataset:
            text = example.get(column, "")
            length = len(text.split())
            lengths.append(length)
            
            if length < min_length:
                too_short += 1
            elif length > max_length:
                too_long += 1
        
        stats = {
            "too_short": too_short,
            "too_long": too_long,
            "valid": len(dataset) - too_short - too_long,
            "avg_length": np.mean(lengths) if lengths else 0,
            "min_length_found": min(lengths) if lengths else 0,
            "max_length_found": max(lengths) if lengths else 0
        }
        
        if too_short > 0:
            logger.warning(f"{too_short} examples are too short (< {min_length} words)")
        if too_long > 0:
            logger.warning(f"{too_long} examples are too long (> {max_length} words)")
        
        logger.info(f"Length validation: {stats['valid']}/{len(dataset)} examples valid")
        
        return stats
    
    def check_language_consistency(
        self,
        dataset: Dataset,
        column: str = "text",
        expected_language: str = "en"
    ) -> Dict:
        """
        Check language consistency (basic check using character patterns).
        
        Args:
            dataset: Dataset to check
            column: Column to check
            expected_language: Expected language code
            
        Returns:
            Dictionary with language consistency statistics
        """
        # Simple heuristic: check for non-ASCII characters
        non_english = 0
        
        for example in dataset:
            text = example.get(column, "")
            # If more than 10% non-ASCII characters, flag as potentially non-English
            non_ascii_ratio = sum(ord(c) > 127 for c in text) / max(len(text), 1)
            if non_ascii_ratio > 0.1:
                non_english += 1
        
        stats = {
            "flagged_as_non_english": non_english,
            "percentage": (non_english / len(dataset)) * 100 if len(dataset) > 0 else 0
        }
        
        if stats["percentage"] > 5:
            logger.warning(
                f"{stats['flagged_as_non_english']} examples may not be in English "
                f"({stats['percentage']:.2f}%)"
            )
        
        return stats
    
    def validate_instruction_format(self, dataset: Dataset) -> Dict:
        """
        Validate instruction-tuning format datasets.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            Dictionary with format validation statistics
        """
        required_fields = ["instruction", "output"]
        issues = []
        
        for idx, example in enumerate(dataset):
            # Check required fields
            for field in required_fields:
                if field not in example or not example[field]:
                    issues.append({
                        "index": idx,
                        "issue": f"Missing or empty {field}"
                    })
            
            # Check if instruction is too similar to output
            if "instruction" in example and "output" in example:
                instruction = example["instruction"].lower()
                output = example["output"].lower()
                if instruction == output:
                    issues.append({
                        "index": idx,
                        "issue": "Instruction identical to output"
                    })
        
        stats = {
            "num_issues": len(issues),
            "percentage": (len(issues) / len(dataset)) * 100 if len(dataset) > 0 else 0,
            "sample_issues": issues[:10]  # First 10 for reference
        }
        
        if stats["num_issues"] > 0:
            logger.warning(f"Found {stats['num_issues']} format issues")
        else:
            logger.info("Instruction format validation passed")
        
        return stats
    
    def check_label_distribution(
        self,
        dataset: Dataset,
        label_column: str
    ) -> Dict:
        """
        Check class label distribution.
        
        Args:
            dataset: Dataset to check
            label_column: Column containing labels
            
        Returns:
            Dictionary with label distribution statistics
        """
        label_counts = {}
        
        for example in dataset:
            label = example.get(label_column)
            label_counts[label] = label_counts.get(label, 0) + 1
        
        total = len(dataset)
        distribution = {
            label: {
                "count": count,
                "percentage": (count / total) * 100
            }
            for label, count in label_counts.items()
        }
        
        # Check for imbalance
        percentages = [info["percentage"] for info in distribution.values()]
        max_ratio = max(percentages) / min(percentages) if min(percentages) > 0 else float('inf')
        
        stats = {
            "distribution": distribution,
            "num_classes": len(label_counts),
            "imbalance_ratio": max_ratio,
            "is_balanced": max_ratio < 3  # Less than 3:1 ratio
        }
        
        logger.info(f"Label distribution: {label_counts}")
        
        if not stats["is_balanced"]:
            logger.warning(f"Dataset is imbalanced (ratio: {max_ratio:.2f}:1)")
        
        return stats
    
    def run_full_validation(
        self,
        dataset: Dataset,
        text_column: str = "text",
        min_length: int = 10,
        max_length: int = 2048
    ) -> Dict:
        """
        Run all validation checks.
        
        Args:
            dataset: Dataset to validate
            text_column: Primary text column
            min_length: Minimum text length
            max_length: Maximum text length
            
        Returns:
            Dictionary with all validation results
        """
        logger.info("Running full dataset validation...")
        
        results = {
            "schema_valid": self.validate_schema(dataset),
            "null_values": self.check_null_values(dataset),
            "duplicates": self.check_duplicates(dataset, text_column),
            "length_validation": self.validate_text_length(
                dataset, text_column, min_length, max_length
            ),
            "language_check": self.check_language_consistency(dataset, text_column)
        }
        
        # Check if it's an instruction dataset
        if "instruction" in dataset.column_names:
            results["instruction_format"] = self.validate_instruction_format(dataset)
        
        # Overall pass/fail
        results["overall_valid"] = (
            results["schema_valid"] and
            results["duplicates"]["num_duplicates"] < len(dataset) * 0.1 and
            results["length_validation"]["valid"] > len(dataset) * 0.9
        )
        
        if results["overall_valid"]:
            logger.info("✓ Dataset passed validation")
        else:
            logger.warning("✗ Dataset has validation issues")
        
        return results
    
    def generate_validation_report(self, validation_results: Dict) -> str:
        """
        Generate human-readable validation report.
        
        Args:
            validation_results: Results from run_full_validation
            
        Returns:
            Formatted report string
        """
        report = ["=" * 60, "DATASET VALIDATION REPORT", "=" * 60, ""]
        
        # Schema
        report.append(f"Schema Valid: {'✓' if validation_results['schema_valid'] else '✗'}")
        report.append("")
        
        # Null values
        report.append("Null Values:")
        for col, info in validation_results['null_values'].items():
            if info['count'] > 0:
                report.append(f"  - {col}: {info['count']} ({info['percentage']:.2f}%)")
        report.append("")
        
        # Duplicates
        dup = validation_results['duplicates']
        report.append(f"Duplicates: {dup['num_duplicates']} ({dup['percentage']:.2f}%)")
        report.append("")
        
        # Length validation
        length = validation_results['length_validation']
        report.append("Length Validation:")
        report.append(f"  - Valid: {length['valid']}")
        report.append(f"  - Too short: {length['too_short']}")
        report.append(f"  - Too long: {length['too_long']}")
        report.append(f"  - Avg length: {length['avg_length']:.1f} words")
        report.append("")
        
        # Overall
        report.append("=" * 60)
        report.append(f"OVERALL: {'PASSED ✓' if validation_results['overall_valid'] else 'FAILED ✗'}")
        report.append("=" * 60)
        
        return "\n".join(report)
