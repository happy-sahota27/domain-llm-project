"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class GenerationRequest(BaseModel):
    """Request model for text generation."""
    
    prompt: str = Field(..., description="Input prompt for generation")
    max_tokens: int = Field(default=256, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: int = Field(default=50, ge=0, description="Top-k sampling parameter")
    stop_sequences: Optional[List[str]] = Field(default=None, description="Stop sequences")
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "Explain the concept of machine learning",
                "max_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50
            }
        }


class GenerationResponse(BaseModel):
    """Response model for text generation."""
    
    generated_text: str = Field(..., description="Generated text")
    prompt: str = Field(..., description="Original prompt")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    finish_reason: str = Field(..., description="Reason for completion (length, stop, etc.)")
    
    class Config:
        schema_extra = {
            "example": {
                "generated_text": "Machine learning is a subset of artificial intelligence...",
                "prompt": "Explain the concept of machine learning",
                "tokens_generated": 125,
                "finish_reason": "length"
            }
        }


class BatchGenerationRequest(BaseModel):
    """Request model for batch text generation."""
    
    prompts: List[str] = Field(..., description="List of input prompts")
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    
    class Config:
        schema_extra = {
            "example": {
                "prompts": [
                    "What is deep learning?",
                    "Explain neural networks."
                ],
                "max_tokens": 200,
                "temperature": 0.7
            }
        }


class BatchGenerationResponse(BaseModel):
    """Response model for batch generation."""
    
    results: List[GenerationResponse] = Field(..., description="List of generation results")
    total_prompts: int = Field(..., description="Total number of prompts processed")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(default=None, description="Name of loaded model")
    version: str = Field(default="1.0.0", description="API version")


class ModelInfo(BaseModel):
    """Model information response."""
    
    model_name: str
    model_type: str
    quantization: Optional[str] = None
    max_context_length: int
    parameters: Optional[str] = None
    domain: Optional[str] = None


class EmbeddingRequest(BaseModel):
    """Request model for embeddings."""
    
    texts: List[str] = Field(..., description="Texts to embed")
    normalize: bool = Field(default=True, description="Normalize embeddings")


class EmbeddingResponse(BaseModel):
    """Response model for embeddings."""
    
    embeddings: List[List[float]] = Field(..., description="Text embeddings")
    dimension: int = Field(..., description="Embedding dimension")


class RerankerRequest(BaseModel):
    """Request for reranking documents."""
    
    query: str = Field(..., description="Search query")
    documents: List[str] = Field(..., description="Documents to rerank")
    top_k: int = Field(default=5, ge=1, description="Number of top documents to return")


class RerankerResponse(BaseModel):
    """Response from reranker."""
    
    results: List[Dict[str, Any]] = Field(..., description="Reranked documents with scores")
    query: str = Field(..., description="Original query")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")
