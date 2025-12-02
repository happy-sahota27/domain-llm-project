"""
Reranker inference module.
"""

import logging
from typing import List, Tuple, Optional, Dict
from sentence_transformers import CrossEncoder
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RerankerInference:
    """Inference interface for reranker models."""
    
    def __init__(self, model_path: str):
        """
        Initialize reranker for inference.
        
        Args:
            model_path: Path to trained reranker model
        """
        self.model_path = model_path
        self.model = None
        
        logger.info(f"Initialized RerankerInference with model: {model_path}")
    
    def load_model(self):
        """Load the reranker model."""
        logger.info("Loading reranker model...")
        
        self.model = CrossEncoder(self.model_path)
        
        logger.info("Model loaded successfully")
    
    def predict(
        self,
        query_doc_pairs: List[Tuple[str, str]],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Predict relevance scores for query-document pairs.
        
        Args:
            query_doc_pairs: List of (query, document) tuples
            batch_size: Batch size for prediction
            
        Returns:
            Array of relevance scores
        """
        if self.model is None:
            self.load_model()
        
        # Format pairs
        pairs = [[query, doc] for query, doc in query_doc_pairs]
        
        # Predict
        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False
        )
        
        return scores
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_scores: bool = True
    ) -> List[Dict]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top documents to return
            return_scores: Whether to include scores in results
            
        Returns:
            List of dictionaries with document, score, and rank
        """
        if self.model is None:
            self.load_model()
        
        # Create pairs
        pairs = [(query, doc) for doc in documents]
        
        # Get scores
        scores = self.predict(pairs)
        
        # Create results
        results = []
        for idx, (doc, score) in enumerate(zip(documents, scores)):
            result = {
                "document": doc,
                "original_index": idx,
                "score": float(score) if return_scores else None
            }
            results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x["score"] if x["score"] is not None else 0, reverse=True)
        
        # Add ranks
        for rank, result in enumerate(results, 1):
            result["rank"] = rank
        
        # Return top_k
        if top_k:
            results = results[:top_k]
        
        logger.info(f"Reranked {len(documents)} documents, returning top {len(results)}")
        
        return results
    
    def batch_rerank(
        self,
        queries: List[str],
        document_lists: List[List[str]],
        top_k: Optional[int] = None
    ) -> List[List[Dict]]:
        """
        Rerank multiple query-document sets.
        
        Args:
            queries: List of queries
            document_lists: List of document lists (one per query)
            top_k: Number of top documents per query
            
        Returns:
            List of reranked results (one per query)
        """
        if len(queries) != len(document_lists):
            raise ValueError("Number of queries must match number of document lists")
        
        results = []
        
        for query, documents in zip(queries, document_lists):
            reranked = self.rerank(query, documents, top_k=top_k)
            results.append(reranked)
        
        return results
    
    def get_top_documents(
        self,
        query: str,
        documents: List[str],
        k: int = 5
    ) -> List[str]:
        """
        Get top-k most relevant documents.
        
        Args:
            query: Search query
            documents: List of documents
            k: Number of top documents to return
            
        Returns:
            List of top-k documents
        """
        results = self.rerank(query, documents, top_k=k, return_scores=False)
        return [r["document"] for r in results]
    
    def compare_documents(
        self,
        query: str,
        doc1: str,
        doc2: str
    ) -> Tuple[str, float]:
        """
        Compare two documents and return the more relevant one.
        
        Args:
            query: Search query
            doc1: First document
            doc2: Second document
            
        Returns:
            Tuple of (more_relevant_document, score_difference)
        """
        pairs = [(query, doc1), (query, doc2)]
        scores = self.predict(pairs)
        
        if scores[0] > scores[1]:
            return doc1, float(scores[0] - scores[1])
        else:
            return doc2, float(scores[1] - scores[0])
    
    def filter_by_threshold(
        self,
        query: str,
        documents: List[str],
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Filter documents by relevance threshold.
        
        Args:
            query: Search query
            documents: List of documents
            threshold: Minimum relevance score
            
        Returns:
            List of documents above threshold with scores
        """
        results = self.rerank(query, documents, return_scores=True)
        
        filtered = [r for r in results if r["score"] >= threshold]
        
        logger.info(f"Filtered {len(documents)} documents to {len(filtered)} above threshold {threshold}")
        
        return filtered
