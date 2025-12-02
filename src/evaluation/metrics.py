"""
Custom evaluation metrics for LLM fine-tuning.
"""

import logging
import numpy as np
import torch
from typing import List, Dict, Optional, Union
from collections import Counter
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate various metrics for LLM evaluation."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        logger.info("Initialized MetricsCalculator")
    
    def calculate_perplexity(
        self,
        model,
        tokenizer,
        texts: List[str],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> float:
        """
        Calculate perplexity on a list of texts.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            texts: List of text strings
            device: Device to use
            
        Returns:
            Average perplexity
        """
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
                input_ids = encodings.input_ids.to(device)
                
                # Get model outputs
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
                
                # Accumulate
                total_loss += loss.item() * input_ids.size(1)
                total_tokens += input_ids.size(1)
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        logger.info(f"Perplexity: {perplexity:.2f}")
        return perplexity
    
    def calculate_accuracy(
        self,
        predictions: List[str],
        references: List[str],
        case_sensitive: bool = False
    ) -> float:
        """
        Calculate exact match accuracy.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            case_sensitive: Whether to consider case
            
        Returns:
            Accuracy score
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        if not case_sensitive:
            predictions = [p.lower().strip() for p in predictions]
            references = [r.lower().strip() for r in references]
        else:
            predictions = [p.strip() for p in predictions]
            references = [r.strip() for r in references]
        
        correct = sum(p == r for p, r in zip(predictions, references))
        accuracy = correct / len(predictions)
        
        logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{len(predictions)})")
        return accuracy
    
    def calculate_token_accuracy(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """
        Calculate token-level accuracy.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            
        Returns:
            Token-level accuracy
        """
        total_tokens = 0
        correct_tokens = 0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            
            # Use the longer length as denominator
            max_len = max(len(pred_tokens), len(ref_tokens))
            total_tokens += max_len
            
            # Count matching tokens at same positions
            for i in range(min(len(pred_tokens), len(ref_tokens))):
                if pred_tokens[i] == ref_tokens[i]:
                    correct_tokens += 1
        
        token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
        logger.info(f"Token Accuracy: {token_accuracy:.4f}")
        return token_accuracy
    
    def calculate_rouge(
        self,
        predictions: List[str],
        references: List[str],
        rouge_types: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            rouge_types: Types of ROUGE to calculate (default: ['rouge1', 'rouge2', 'rougeL'])
            
        Returns:
            Dictionary of ROUGE scores
        """
        try:
            from rouge_score import rouge_scorer
            
            if rouge_types is None:
                rouge_types = ['rouge1', 'rouge2', 'rougeL']
            
            scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
            
            scores = {rouge_type: [] for rouge_type in rouge_types}
            
            for pred, ref in zip(predictions, references):
                rouge_score = scorer.score(ref, pred)
                for rouge_type in rouge_types:
                    scores[rouge_type].append(rouge_score[rouge_type].fmeasure)
            
            # Average scores
            avg_scores = {
                rouge_type: np.mean(score_list)
                for rouge_type, score_list in scores.items()
            }
            
            for rouge_type, score in avg_scores.items():
                logger.info(f"{rouge_type}: {score:.4f}")
            
            return avg_scores
        
        except ImportError:
            logger.warning("rouge_score not installed. Using basic ROUGE implementation.")
            return self._calculate_rouge_basic(predictions, references)
    
    def _calculate_rouge_basic(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Basic ROUGE-1 implementation without external dependencies."""
        rouge1_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            
            if len(ref_tokens) == 0:
                continue
            
            overlap = len(pred_tokens & ref_tokens)
            precision = overlap / len(pred_tokens) if len(pred_tokens) > 0 else 0
            recall = overlap / len(ref_tokens)
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            
            rouge1_scores.append(f1)
        
        return {"rouge1": np.mean(rouge1_scores) if rouge1_scores else 0}
    
    def calculate_bleu(
        self,
        predictions: List[str],
        references: List[List[str]],
        max_n: int = 4
    ) -> Dict[str, float]:
        """
        Calculate BLEU scores.
        
        Args:
            predictions: List of predicted strings
            references: List of reference lists (multiple references per prediction)
            max_n: Maximum n-gram order
            
        Returns:
            Dictionary of BLEU scores
        """
        try:
            from sacrebleu import corpus_bleu
            
            # Convert references format
            # sacrebleu expects list of lists where each sublist contains references for one prediction
            if not isinstance(references[0], list):
                references = [[ref] for ref in references]
            
            # Transpose references for sacrebleu format
            refs_transposed = list(zip(*references))
            
            bleu = corpus_bleu(predictions, refs_transposed)
            
            result = {
                "bleu": bleu.score / 100,  # Normalize to 0-1
                "bleu_precisions": bleu.precisions
            }
            
            logger.info(f"BLEU: {result['bleu']:.4f}")
            return result
        
        except ImportError:
            logger.warning("sacrebleu not installed. Using basic BLEU implementation.")
            return self._calculate_bleu_basic(predictions, references)
    
    def _calculate_bleu_basic(
        self,
        predictions: List[str],
        references: List[List[str]]
    ) -> Dict[str, float]:
        """Basic BLEU implementation without external dependencies."""
        from collections import Counter
        import math
        
        def get_ngrams(tokens: List[str], n: int) -> Counter:
            """Extract n-grams from tokens."""
            return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
        
        bleu_scores = []
        
        for pred, refs in zip(predictions, references):
            pred_tokens = pred.lower().split()
            
            if not isinstance(refs, list):
                refs = [refs]
            
            # Calculate precision for each n-gram order
            precisions = []
            for n in range(1, 5):  # 1-gram to 4-gram
                pred_ngrams = get_ngrams(pred_tokens, n)
                
                max_overlap = 0
                for ref in refs:
                    ref_tokens = ref.lower().split()
                    ref_ngrams = get_ngrams(ref_tokens, n)
                    
                    overlap = sum((pred_ngrams & ref_ngrams).values())
                    max_overlap = max(max_overlap, overlap)
                
                precision = max_overlap / len(pred_tokens) if len(pred_tokens) >= n else 0
                precisions.append(precision)
            
            # Calculate geometric mean of precisions
            if all(p > 0 for p in precisions):
                bleu = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
            else:
                bleu = 0
            
            bleu_scores.append(bleu)
        
        return {"bleu": np.mean(bleu_scores) if bleu_scores else 0}
    
    def calculate_f1(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Calculate F1 score based on token overlap.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            
        Returns:
            Dictionary with precision, recall, and F1
        """
        precisions = []
        recalls = []
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            
            if len(pred_tokens) == 0 and len(ref_tokens) == 0:
                precisions.append(1.0)
                recalls.append(1.0)
                f1_scores.append(1.0)
                continue
            
            overlap = len(pred_tokens & ref_tokens)
            
            precision = overlap / len(pred_tokens) if len(pred_tokens) > 0 else 0
            recall = overlap / len(ref_tokens) if len(ref_tokens) > 0 else 0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        result = {
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "f1": np.mean(f1_scores)
        }
        
        logger.info(f"Precision: {result['precision']:.4f}, Recall: {result['recall']:.4f}, F1: {result['f1']:.4f}")
        return result
    
    def calculate_semantic_similarity(
        self,
        predictions: List[str],
        references: List[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> float:
        """
        Calculate semantic similarity using sentence embeddings.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            model_name: Sentence transformer model name
            
        Returns:
            Average cosine similarity
        """
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            
            model = SentenceTransformer(model_name)
            
            # Encode sentences
            pred_embeddings = model.encode(predictions)
            ref_embeddings = model.encode(references)
            
            # Calculate cosine similarities
            similarities = []
            for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
                sim = cosine_similarity([pred_emb], [ref_emb])[0][0]
                similarities.append(sim)
            
            avg_similarity = np.mean(similarities)
            logger.info(f"Semantic Similarity: {avg_similarity:.4f}")
            return avg_similarity
        
        except ImportError:
            logger.warning("sentence-transformers not installed. Skipping semantic similarity.")
            return 0.0
    
    def calculate_all_metrics(
        self,
        predictions: List[str],
        references: List[str],
        model=None,
        tokenizer=None,
        include_perplexity: bool = False
    ) -> Dict[str, Union[float, Dict]]:
        """
        Calculate all available metrics.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            model: Optional model for perplexity
            tokenizer: Optional tokenizer for perplexity
            include_perplexity: Whether to calculate perplexity
            
        Returns:
            Dictionary of all metrics
        """
        logger.info("Calculating all metrics...")
        
        metrics = {}
        
        # Accuracy
        metrics["accuracy"] = self.calculate_accuracy(predictions, references)
        
        # Token accuracy
        metrics["token_accuracy"] = self.calculate_token_accuracy(predictions, references)
        
        # F1
        metrics.update(self.calculate_f1(predictions, references))
        
        # ROUGE
        rouge_scores = self.calculate_rouge(predictions, references)
        metrics.update(rouge_scores)
        
        # BLEU
        refs_list = [[ref] for ref in references]
        bleu_scores = self.calculate_bleu(predictions, refs_list)
        metrics.update(bleu_scores)
        
        # Perplexity (if model provided)
        if include_perplexity and model is not None and tokenizer is not None:
            metrics["perplexity"] = self.calculate_perplexity(model, tokenizer, references)
        
        logger.info("All metrics calculated successfully")
        return metrics
