"""
Reranker model trainer for document reranking.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional
import torch
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryAccuracyEvaluator
from torch.utils.data import DataLoader
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RerankerTrainer:
    """Train cross-encoder models for document reranking."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        output_dir: str = "models/reranker"
    ):
        """
        Initialize reranker trainer.
        
        Args:
            model_name: Base model to fine-tune
            output_dir: Directory to save trained model
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        
        logger.info(f"Initialized RerankerTrainer with model: {model_name}")
    
    def load_model(self, num_labels: int = 1):
        """
        Load cross-encoder model.
        
        Args:
            num_labels: Number of output labels (1 for regression, >1 for classification)
        """
        logger.info("Loading cross-encoder model...")
        
        self.model = CrossEncoder(
            self.model_name,
            num_labels=num_labels,
            max_length=512
        )
        
        logger.info("Model loaded successfully")
    
    def prepare_training_data(
        self,
        query_doc_pairs: List[Tuple[str, str]],
        labels: List[float]
    ) -> List[InputExample]:
        """
        Prepare training data in the format required by sentence-transformers.
        
        Args:
            query_doc_pairs: List of (query, document) tuples
            labels: List of relevance labels (0-1 for binary, 0-N for ranking)
            
        Returns:
            List of InputExample objects
        """
        if len(query_doc_pairs) != len(labels):
            raise ValueError("Number of pairs must match number of labels")
        
        examples = []
        for (query, doc), label in zip(query_doc_pairs, labels):
            examples.append(InputExample(texts=[query, doc], label=float(label)))
        
        logger.info(f"Prepared {len(examples)} training examples")
        return examples
    
    def train(
        self,
        train_examples: List[InputExample],
        eval_examples: Optional[List[InputExample]] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100,
        evaluation_steps: int = 1000,
        save_best_model: bool = True
    ):
        """
        Train the reranker model.
        
        Args:
            train_examples: Training examples
            eval_examples: Optional evaluation examples
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            evaluation_steps: Steps between evaluations
            save_best_model: Whether to save the best model
        """
        if self.model is None:
            self.load_model()
        
        logger.info(f"Starting training with {len(train_examples)} examples...")
        
        # Create data loader
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=batch_size
        )
        
        # Set up evaluator if eval data provided
        evaluator = None
        if eval_examples:
            evaluator = CEBinaryAccuracyEvaluator.from_input_examples(
                eval_examples,
                name="eval"
            )
        
        # Train
        self.model.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=epochs,
            evaluation_steps=evaluation_steps,
            warmup_steps=warmup_steps,
            output_path=str(self.output_dir),
            save_best_model=save_best_model,
            show_progress_bar=True
        )
        
        logger.info(f"Training completed. Model saved to {self.output_dir}")
    
    def predict(
        self,
        query_doc_pairs: List[Tuple[str, str]],
        batch_size: int = 32
    ) -> List[float]:
        """
        Predict relevance scores for query-document pairs.
        
        Args:
            query_doc_pairs: List of (query, document) tuples
            batch_size: Batch size for prediction
            
        Returns:
            List of relevance scores
        """
        if self.model is None:
            raise ValueError("Model must be loaded or trained before prediction")
        
        # Format pairs as list of lists
        pairs = [[query, doc] for query, doc in query_doc_pairs]
        
        # Predict
        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=True
        )
        
        return scores.tolist() if hasattr(scores, 'tolist') else scores
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float, int]]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top documents to return (None = all)
            
        Returns:
            List of (document, score, original_index) tuples, sorted by score
        """
        # Create query-document pairs
        pairs = [(query, doc) for doc in documents]
        
        # Get scores
        scores = self.predict(pairs)
        
        # Combine with documents and indices
        results = [
            (doc, score, idx)
            for idx, (doc, score) in enumerate(zip(documents, scores))
        ]
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k if specified
        if top_k:
            results = results[:top_k]
        
        return results
    
    def evaluate(
        self,
        eval_examples: List[InputExample],
        batch_size: int = 32
    ) -> dict:
        """
        Evaluate the model on test data.
        
        Args:
            eval_examples: Evaluation examples
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be loaded before evaluation")
        
        logger.info(f"Evaluating on {len(eval_examples)} examples...")
        
        evaluator = CEBinaryAccuracyEvaluator.from_input_examples(
            eval_examples,
            name="test"
        )
        
        score = evaluator(self.model, output_path=str(self.output_dir))
        
        metrics = {"accuracy": score}
        
        logger.info(f"Evaluation accuracy: {score:.4f}")
        
        # Save metrics
        metrics_file = self.output_dir / "eval_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def save_model(self, path: Optional[str] = None):
        """
        Save the trained model.
        
        Args:
            path: Optional custom save path
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        save_path = path or str(self.output_dir)
        self.model.save(save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_trained_model(self, path: str):
        """
        Load a previously trained model.
        
        Args:
            path: Path to saved model
        """
        logger.info(f"Loading trained model from {path}")
        
        self.model = CrossEncoder(path)
        
        logger.info("Model loaded successfully")
    
    @staticmethod
    def create_synthetic_dataset(
        queries: List[str],
        documents: List[str],
        num_samples: int = 1000
    ) -> Tuple[List[Tuple[str, str]], List[float]]:
        """
        Create a synthetic training dataset.
        
        Args:
            queries: List of queries
            documents: List of documents
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (pairs, labels)
        """
        import random
        
        pairs = []
        labels = []
        
        for _ in range(num_samples):
            query = random.choice(queries)
            doc = random.choice(documents)
            
            # Simple heuristic: relevant if they share words
            query_words = set(query.lower().split())
            doc_words = set(doc.lower().split())
            
            overlap = len(query_words & doc_words)
            
            # Label: 1 if overlap > 2, else 0
            label = 1.0 if overlap > 2 else 0.0
            
            pairs.append((query, doc))
            labels.append(label)
        
        logger.info(f"Created {len(pairs)} synthetic training examples")
        return pairs, labels
