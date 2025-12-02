"""
Unit tests for evaluation metrics.
"""

import pytest
from src.evaluation.metrics import MetricsCalculator


def test_metrics_calculator():
    """Test MetricsCalculator initialization."""
    calculator = MetricsCalculator()
    assert calculator is not None


def test_accuracy():
    """Test accuracy calculation."""
    calculator = MetricsCalculator()
    
    predictions = ["hello world", "test case", "example"]
    references = ["hello world", "test case", "sample"]
    
    accuracy = calculator.calculate_accuracy(predictions, references)
    
    assert accuracy == 2/3


def test_case_insensitive_accuracy():
    """Test case-insensitive accuracy."""
    calculator = MetricsCalculator()
    
    predictions = ["Hello World", "TEST CASE"]
    references = ["hello world", "test case"]
    
    accuracy = calculator.calculate_accuracy(predictions, references, case_sensitive=False)
    
    assert accuracy == 1.0


def test_token_accuracy():
    """Test token-level accuracy."""
    calculator = MetricsCalculator()
    
    predictions = ["hello world", "test case"]
    references = ["hello world", "test case"]
    
    accuracy = calculator.calculate_token_accuracy(predictions, references)
    
    assert accuracy == 1.0


def test_f1_score():
    """Test F1 score calculation."""
    calculator = MetricsCalculator()
    
    predictions = ["the cat sat on the mat"]
    references = ["the cat sat on the mat"]
    
    result = calculator.calculate_f1(predictions, references)
    
    assert result["f1"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_partial_overlap_f1():
    """Test F1 with partial overlap."""
    calculator = MetricsCalculator()
    
    predictions = ["the cat sat"]
    references = ["the dog sat"]
    
    result = calculator.calculate_f1(predictions, references)
    
    # Should have non-zero F1 due to overlap on "the" and "sat"
    assert 0 < result["f1"] < 1.0


def test_rouge_basic():
    """Test basic ROUGE calculation."""
    calculator = MetricsCalculator()
    
    predictions = ["the cat sat on the mat"]
    references = ["the cat sat on the mat"]
    
    rouge_scores = calculator._calculate_rouge_basic(predictions, references)
    
    assert "rouge1" in rouge_scores
    assert rouge_scores["rouge1"] == 1.0


def test_bleu_basic():
    """Test basic BLEU calculation."""
    calculator = MetricsCalculator()
    
    predictions = ["the cat sat on the mat"]
    references = [["the cat sat on the mat"]]
    
    bleu_scores = calculator._calculate_bleu_basic(predictions, references)
    
    assert "bleu" in bleu_scores
    assert bleu_scores["bleu"] > 0


def test_all_metrics():
    """Test calculating all metrics together."""
    calculator = MetricsCalculator()
    
    predictions = ["hello world", "test case"]
    references = ["hello world", "test case"]
    
    metrics = calculator.calculate_all_metrics(
        predictions=predictions,
        references=references,
        include_perplexity=False
    )
    
    assert "accuracy" in metrics
    assert "token_accuracy" in metrics
    assert "f1" in metrics
    assert "rouge1" in metrics
    assert "bleu" in metrics
    
    # Perfect match should give high scores
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
