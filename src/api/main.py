"""
FastAPI application for model deployment.
"""

import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time
from contextlib import asynccontextmanager

from .routes import router, set_model, set_reranker
from .models import ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model holders
app_model = None
app_tokenizer = None
app_model_name = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for app startup/shutdown.
    """
    # Startup
    logger.info("Starting up API server...")
    logger.info("Model will be loaded on first request or via load_model()")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server...")
    global app_model, app_tokenizer
    app_model = None
    app_tokenizer = None
    logger.info("Cleanup completed")


# Create FastAPI app
app = FastAPI(
    title="Domain LLM API",
    description="API for fine-tuned domain-specific language models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()
    
    logger.info(f"Request: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    logger.info(f"Response: {response.status_code} (took {duration:.2f}s)")
    
    return response


# Exception handlers
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "status_code": 500
        }
    )


# Include routers
app.include_router(router, prefix="/api/v1", tags=["inference"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Domain LLM API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


def load_model(model, tokenizer, model_name: str):
    """
    Load model into the API.
    
    Args:
        model: The model instance
        tokenizer: The tokenizer instance
        model_name: Name of the model
    """
    global app_model, app_tokenizer, app_model_name
    
    app_model = model
    app_tokenizer = tokenizer
    app_model_name = model_name
    
    # Set in routes
    set_model(model, tokenizer, model_name)
    
    logger.info(f"Model '{model_name}' loaded successfully")


def load_reranker(reranker):
    """
    Load reranker model into the API.
    
    Args:
        reranker: The reranker model instance
    """
    set_reranker(reranker)
    logger.info("Reranker loaded successfully")


def load_quantized_model(model_path: str, n_ctx: int = 2048):
    """
    Load a quantized GGUF model.
    
    Args:
        model_path: Path to GGUF model file
        n_ctx: Context window size
    """
    try:
        from llama_cpp import Llama
        
        logger.info(f"Loading quantized model from {model_path}")
        
        model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=None,  # Use all CPU threads
            n_gpu_layers=-1,  # Use GPU if available
            verbose=False
        )
        
        # Create a simple tokenizer wrapper
        class SimpleTokenizer:
            def __init__(self, llama_model):
                self.model = llama_model
                self.eos_token_id = 2  # Common EOS token ID
            
            def __call__(self, text, return_tensors=None, **kwargs):
                # Return a dummy dict for compatibility
                tokens = self.model.tokenize(text.encode())
                return {"input_ids": [[len(tokens)]]}
            
            def decode(self, tokens, **kwargs):
                if hasattr(tokens, 'tolist'):
                    tokens = tokens.tolist()
                return self.model.detokenize(tokens).decode()
        
        tokenizer = SimpleTokenizer(model)
        
        load_model(model, tokenizer, model_path)
        
        logger.info("Quantized model loaded successfully")
        
    except ImportError:
        logger.error("llama-cpp-python not installed")
        raise
    except Exception as e:
        logger.error(f"Failed to load quantized model: {e}")
        raise


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
