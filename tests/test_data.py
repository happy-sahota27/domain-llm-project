"""
Unit tests for dataset handling.
"""

import pytest
from src.data.dataset_builder import DomainDatasetBuilder
from src.data.preprocessing import DataPreprocessor
from src.data.validation import DataValidator


def test_domain_dataset_builder():
    """Test DomainDatasetBuilder initialization."""
    builder = DomainDatasetBuilder(domain="healthcare")
    assert builder.domain == "healthcare"
    assert builder.output_dir.name == "processed"


def test_create_sample_dataset():
    """Test sample dataset creation."""
    builder = DomainDatasetBuilder(domain="healthcare")
    dataset = builder.create_sample_dataset(num_samples=10)
    
    assert len(dataset) == 10
    assert "instruction" in dataset.column_names
    assert "input" in dataset.column_names
    assert "output" in dataset.column_names


def test_instruction_format():
    """Test instruction dataset formatting."""
    builder = DomainDatasetBuilder(domain="healthcare")
    
    data = [
        {
            "instruction": "Explain the condition",
            "input": "What is diabetes?",
            "output": "Diabetes is a metabolic disease."
        }
    ]
    
    dataset = builder.create_instruction_dataset(data)
    
    assert len(dataset) == 1
    assert "text" in dataset.column_names
    assert "### Instruction:" in dataset[0]["text"]


def test_dataset_split():
    """Test dataset splitting."""
    builder = DomainDatasetBuilder(domain="healthcare")
    dataset = builder.create_sample_dataset(num_samples=100)
    
    split_dataset = builder.split_dataset(
        dataset,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1
    )
    
    assert "train" in split_dataset
    assert "validation" in split_dataset
    assert "test" in split_dataset
    assert len(split_dataset["train"]) == 80
    assert len(split_dataset["validation"]) == 10
    assert len(split_dataset["test"]) == 10


def test_data_preprocessor():
    """Test DataPreprocessor."""
    preprocessor = DataPreprocessor(max_length=100, min_length=5)
    
    text = "  This   is   a   test.  "
    cleaned = preprocessor.clean_text(text)
    
    assert cleaned == "This is a test."


def test_remove_special_characters():
    """Test special character removal."""
    preprocessor = DataPreprocessor()
    
    text = "Hello @world! #test"
    cleaned = preprocessor.remove_special_characters(text, keep_punctuation=True)
    
    assert "@" not in cleaned
    assert "#" not in cleaned
    assert "!" in cleaned


def test_data_validator():
    """Test DataValidator."""
    from datasets import Dataset
    
    validator = DataValidator(required_columns=["text"])
    
    # Valid dataset
    dataset = Dataset.from_dict({"text": ["Hello", "World"]})
    assert validator.validate_schema(dataset) == True
    
    # Invalid dataset
    dataset_invalid = Dataset.from_dict({"content": ["Hello", "World"]})
    assert validator.validate_schema(dataset_invalid) == False


def test_null_value_check():
    """Test null value detection."""
    from datasets import Dataset
    
    validator = DataValidator()
    dataset = Dataset.from_dict({
        "text": ["Hello", "", "World", None]
    })
    
    null_counts = validator.check_null_values(dataset, columns=["text"])
    
    assert null_counts["text"]["count"] == 2


def test_duplicate_check():
    """Test duplicate detection."""
    from datasets import Dataset
    
    validator = DataValidator()
    dataset = Dataset.from_dict({
        "text": ["Hello", "World", "Hello", "Test"]
    })
    
    dup_stats = validator.check_duplicates(dataset, column="text")
    
    assert dup_stats["num_duplicates"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
