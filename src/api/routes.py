"""
API routes for model inference.
"""

import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional
import time

from .models import (
    GenerationRequest,
    GenerationResponse,
    BatchGenerationRequest,
    BatchGenerationResponse,
    HealthResponse,
    ModelInfo,
    RerankerRequest,
    RerankerResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Global model holder (will be set during app startup)
model = None
tokenizer = None
model_name = None
reranker = None


def set_model(m, t, name: str):
    """Set the global model, tokenizer, and name."""
    global model, tokenizer, model_name
    model = m
    tokenizer = t
    model_name = name


def set_reranker(r):
    """Set the global reranker model."""
    global reranker
    reranker = r


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Service health status
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_name=model_name,
        version="1.0.0"
    )


@router.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """
    Get information about the loaded model.
    
    Returns:
        Model information
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_name=model_name or "unknown",
        model_type="causal-lm",
        quantization="GGUF" if "gguf" in str(model_name).lower() else None,
        max_context_length=2048,
        parameters="7B",
        domain="multi-domain"
    )


@router.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """
    Generate text from a prompt.
    
    Args:
        request: Generation request
        
    Returns:
        Generated text response
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Generating text for prompt: {request.prompt[:50]}...")
        
        start_time = time.time()
        
        # Tokenize input
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        input_length = inputs.input_ids.shape[1]
        
        # Move to model device
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        tokens_generated = len(generated_tokens)
        
        elapsed = time.time() - start_time
        logger.info(f"Generation completed in {elapsed:.2f}s ({tokens_generated} tokens)")
        
        return GenerationResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            tokens_generated=tokens_generated,
            finish_reason="length" if tokens_generated >= request.max_tokens else "stop"
        )
    
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/generate/batch", response_model=BatchGenerationResponse)
async def generate_batch(request: BatchGenerationRequest):
    """
    Generate text for multiple prompts.
    
    Args:
        request: Batch generation request
        
    Returns:
        Batch generation responses
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Batch generation for {len(request.prompts)} prompts")
        
        results = []
        
        for prompt in request.prompts:
            # Create individual request
            individual_request = GenerationRequest(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            )
            
            # Generate
            response = await generate_text(individual_request)
            results.append(response)
        
        return BatchGenerationResponse(
            results=results,
            total_prompts=len(request.prompts)
        )
    
    except Exception as e:
        logger.error(f"Batch generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")


@router.post("/rerank", response_model=RerankerResponse)
async def rerank_documents(request: RerankerRequest):
    """
    Rerank documents based on relevance to query.
    
    Args:
        request: Reranker request
        
    Returns:
        Reranked documents with scores
    """
    if reranker is None:
        raise HTTPException(
            status_code=503,
            detail="Reranker not loaded. Use /generate endpoint instead."
        )
    
    try:
        logger.info(f"Reranking {len(request.documents)} documents for query: {request.query[:50]}...")
        
        # Create pairs of (query, document)
        pairs = [(request.query, doc) for doc in request.documents]
        
        # Get scores from reranker
        scores = reranker.predict(pairs)
        
        # Combine documents with scores
        results = [
            {
                "document": doc,
                "score": float(score),
                "rank": i + 1
            }
            for i, (doc, score) in enumerate(zip(request.documents, scores))
        ]
        
        # Sort by score (descending)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top_k
        results = results[:request.top_k]
        
        # Update ranks
        for i, result in enumerate(results):
            result["rank"] = i + 1
        
        return RerankerResponse(
            results=results,
            query=request.query
        )
    
    except Exception as e:
        logger.error(f"Reranking error: {e}")
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")


@router.post("/shutdown")
async def shutdown(background_tasks: BackgroundTasks):
    """
    Gracefully shutdown the server.
    
    Returns:
        Shutdown confirmation
    """
    logger.info("Shutdown requested")
    
    def cleanup():
        global model, tokenizer, reranker
        model = None
        tokenizer = None
        reranker = None
        logger.info("Cleanup completed")
    
    background_tasks.add_task(cleanup)
    
    return {"message": "Shutdown initiated"}
