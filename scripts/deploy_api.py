#!/usr/bin/env python3
"""
Script to deploy model API.
"""

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from src.api.main import app, load_model, load_quantized_model, load_reranker
from src.training.qlora_trainer import QLoRATrainer
from src.training.config import TrainingConfig
from src.reranker.inference import RerankerInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Deploy model API")
    
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to trained model (HuggingFace format)"
    )
    parser.add_argument(
        "--quantized-model",
        type=str,
        help="Path to quantized GGUF model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Base model name (for LoRA models)"
    )
    parser.add_argument(
        "--reranker-path",
        type=str,
        help="Path to trained reranker model"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only)"
    )
    
    args = parser.parse_args()
    
    # Load model
    if args.quantized_model:
        logger.info(f"Loading quantized model: {args.quantized_model}")
        load_quantized_model(args.quantized_model)
    
    elif args.model_path:
        logger.info(f"Loading model: {args.model_path}")
        config = TrainingConfig(model_name=args.base_model)
        trainer = QLoRATrainer(config)
        trainer.load_trained_model(args.model_path)
        load_model(trainer.model, trainer.tokenizer, args.model_path)
    
    else:
        logger.warning("No model specified. API will start but model endpoints will return 503")
    
    # Load reranker if specified
    if args.reranker_path:
        logger.info(f"Loading reranker: {args.reranker_path}")
        reranker = RerankerInference(args.reranker_path)
        reranker.load_model()
        load_reranker(reranker)
    
    # Start server
    logger.info("="*60)
    logger.info(f"Starting API server on {args.host}:{args.port}")
    logger.info(f"Docs available at: http://{args.host}:{args.port}/docs")
    logger.info("="*60)
    
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
