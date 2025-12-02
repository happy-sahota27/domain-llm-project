"""
Model evaluation framework for comprehensive testing.
"""

import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
import torch
from datasets import Dataset
from tqdm import tqdm

from .metrics import MetricsCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation framework."""
    
    def __init__(
        self,
        model,
        tokenizer,
        metrics_calculator: Optional[MetricsCalculator] = None,
        output_dir: str = "results/evaluation"
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            metrics_calculator: Optional MetricsCalculator instance
            output_dir: Directory to save results
        """
        self.model = model
        self.tokenizer = tokenizer
        self.metrics_calculator = metrics_calculator or MetricsCalculator()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if hasattr(model, 'device'):
            self.device = model.device
        
        logger.info(f"Initialized ModelEvaluator on device: {self.device}")
    
    def generate_predictions(
        self,
        dataset: Dataset,
        input_column: str = "input",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        batch_size: int = 1
    ) -> List[str]:
        """
        Generate predictions for a dataset.
        
        Args:
            dataset: Evaluation dataset
            input_column: Column containing input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            batch_size: Batch size for generation
            
        Returns:
            List of generated predictions
        """
        logger.info(f"Generating predictions for {len(dataset)} examples...")
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(dataset), batch_size), desc="Generating"):
                batch = dataset[i:i+batch_size]
                
                if isinstance(batch[input_column], str):
                    inputs = [batch[input_column]]
                else:
                    inputs = batch[input_column]
                
                # Tokenize
                encodings = self.tokenizer(
                    inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                ).to(self.device)
                
                # Generate
                outputs = self.model.generate(
                    **encodings,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode
                for output in outputs:
                    # Remove input tokens from output
                    generated_tokens = output[encodings.input_ids.shape[1]:]
                    prediction = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    predictions.append(prediction)
        
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions
    
    def evaluate_dataset(
        self,
        dataset: Dataset,
        input_column: str = "input",
        reference_column: str = "output",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        include_perplexity: bool = True,
        save_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataset: Evaluation dataset
            input_column: Column with inputs
            reference_column: Column with reference outputs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            include_perplexity: Whether to calculate perplexity
            save_predictions: Whether to save predictions to file
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("=" * 60)
        logger.info("STARTING EVALUATION")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Generate predictions
        predictions = self.generate_predictions(
            dataset,
            input_column=input_column,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        # Get references
        references = dataset[reference_column]
        if isinstance(references, str):
            references = [references]
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_all_metrics(
            predictions=predictions,
            references=references,
            model=self.model if include_perplexity else None,
            tokenizer=self.tokenizer if include_perplexity else None,
            include_perplexity=include_perplexity
        )
        
        # Add timing info
        eval_time = time.time() - start_time
        metrics["evaluation_time_seconds"] = eval_time
        metrics["examples_per_second"] = len(dataset) / eval_time
        
        # Save results
        if save_predictions:
            self._save_predictions(predictions, references, dataset[input_column], metrics)
        
        self._save_metrics(metrics)
        
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        self._print_metrics(metrics)
        
        return metrics
    
    def compare_models(
        self,
        models: Dict[str, Any],
        tokenizers: Dict[str, Any],
        dataset: Dataset,
        input_column: str = "input",
        reference_column: str = "output"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models on the same dataset.
        
        Args:
            models: Dictionary of model_name -> model
            tokenizers: Dictionary of model_name -> tokenizer
            dataset: Evaluation dataset
            input_column: Column with inputs
            reference_column: Column with reference outputs
            
        Returns:
            Dictionary of model_name -> metrics
        """
        logger.info(f"Comparing {len(models)} models...")
        
        results = {}
        
        for model_name in models:
            logger.info(f"\nEvaluating model: {model_name}")
            logger.info("-" * 60)
            
            # Update model and tokenizer
            self.model = models[model_name]
            self.tokenizer = tokenizers[model_name]
            
            # Evaluate
            metrics = self.evaluate_dataset(
                dataset,
                input_column=input_column,
                reference_column=reference_column,
                save_predictions=False
            )
            
            results[model_name] = metrics
        
        # Save comparison
        self._save_comparison(results)
        self._print_comparison(results)
        
        return results
    
    def evaluate_by_domain(
        self,
        dataset: Dataset,
        domain_column: str,
        input_column: str = "input",
        reference_column: str = "output"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate model separately for different domains.
        
        Args:
            dataset: Dataset with domain labels
            domain_column: Column containing domain labels
            input_column: Column with inputs
            reference_column: Column with references
            
        Returns:
            Dictionary of domain -> metrics
        """
        # Get unique domains
        domains = set(dataset[domain_column])
        logger.info(f"Evaluating across {len(domains)} domains: {domains}")
        
        results = {}
        
        for domain in domains:
            logger.info(f"\nEvaluating domain: {domain}")
            logger.info("-" * 60)
            
            # Filter dataset for this domain
            domain_dataset = dataset.filter(lambda x: x[domain_column] == domain)
            
            # Evaluate
            metrics = self.evaluate_dataset(
                domain_dataset,
                input_column=input_column,
                reference_column=reference_column,
                save_predictions=False
            )
            
            results[domain] = metrics
        
        # Save domain comparison
        self._save_domain_comparison(results)
        
        return results
    
    def benchmark_inference_speed(
        self,
        dataset: Dataset,
        input_column: str = "input",
        num_runs: int = 3,
        max_new_tokens: int = 256
    ) -> Dict[str, float]:
        """
        Benchmark model inference speed.
        
        Args:
            dataset: Test dataset
            input_column: Column with inputs
            num_runs: Number of benchmark runs
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with speed metrics
        """
        logger.info(f"Benchmarking inference speed over {num_runs} runs...")
        
        times = []
        tokens_generated = []
        
        for run in range(num_runs):
            logger.info(f"Run {run + 1}/{num_runs}")
            
            start_time = time.time()
            predictions = self.generate_predictions(
                dataset,
                input_column=input_column,
                max_new_tokens=max_new_tokens,
                temperature=0.7
            )
            elapsed = time.time() - start_time
            
            times.append(elapsed)
            
            # Count tokens
            total_tokens = sum(len(self.tokenizer.encode(pred)) for pred in predictions)
            tokens_generated.append(total_tokens)
        
        metrics = {
            "avg_time_seconds": sum(times) / len(times),
            "min_time_seconds": min(times),
            "max_time_seconds": max(times),
            "avg_examples_per_second": len(dataset) / (sum(times) / len(times)),
            "avg_tokens_per_second": sum(tokens_generated) / sum(times)
        }
        
        logger.info("Benchmark Results:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.2f}")
        
        return metrics
    
    def _save_predictions(
        self,
        predictions: List[str],
        references: List[str],
        inputs: List[str],
        metrics: Dict[str, Any]
    ):
        """Save predictions to JSON file."""
        output_file = self.output_dir / "predictions.json"
        
        data = {
            "metrics": metrics,
            "examples": [
                {
                    "input": inp,
                    "prediction": pred,
                    "reference": ref
                }
                for inp, pred, ref in zip(inputs, predictions, references)
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Predictions saved to {output_file}")
    
    def _save_metrics(self, metrics: Dict[str, Any]):
        """Save metrics to JSON file."""
        output_file = self.output_dir / "metrics.json"
        
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"Metrics saved to {output_file}")
    
    def _save_comparison(self, results: Dict[str, Dict[str, Any]]):
        """Save model comparison results."""
        output_file = self.output_dir / "model_comparison.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Comparison saved to {output_file}")
    
    def _save_domain_comparison(self, results: Dict[str, Dict[str, Any]]):
        """Save domain-wise evaluation results."""
        output_file = self.output_dir / "domain_evaluation.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Domain evaluation saved to {output_file}")
    
    def _print_metrics(self, metrics: Dict[str, Any]):
        """Print metrics in a formatted way."""
        logger.info("\nEvaluation Metrics:")
        logger.info("-" * 60)
        
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            elif isinstance(value, dict):
                logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    logger.info(f"    {sub_key}: {sub_value}")
            else:
                logger.info(f"  {key}: {value}")
    
    def _print_comparison(self, results: Dict[str, Dict[str, Any]]):
        """Print model comparison results."""
        logger.info("\n" + "=" * 60)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 60)
        
        # Get all metric keys
        all_keys = set()
        for metrics in results.values():
            all_keys.update(metrics.keys())
        
        # Print comparison table
        for key in sorted(all_keys):
            if key in ["evaluation_time_seconds", "examples_per_second"]:
                continue
            
            logger.info(f"\n{key}:")
            for model_name, metrics in results.items():
                value = metrics.get(key, "N/A")
                if isinstance(value, float):
                    logger.info(f"  {model_name}: {value:.4f}")
                else:
                    logger.info(f"  {model_name}: {value}")
