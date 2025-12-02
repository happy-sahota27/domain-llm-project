#!/usr/bin/env python3
"""
Script to quantize models to GGUF format.
"""

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quantization.gguf_converter import GGUFConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Quantize model to GGUF format")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to HuggingFace model to quantize"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/quantized",
        help="Output directory for quantized models"
    )
    parser.add_argument(
        "--quantization-types",
        nargs="+",
        default=["q4_k_m", "q5_k_m", "q8_0"],
        help="Quantization types to create"
    )
    parser.add_argument(
        "--llama-cpp-path",
        type=str,
        help="Path to llama.cpp directory"
    )
    parser.add_argument(
        "--vocab-type",
        type=str,
        default="spm",
        choices=["spm", "bpe", "hfft"],
        help="Vocabulary type"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run quantization benchmarks"
    )
    parser.add_argument(
        "--test-generation",
        action="store_true",
        help="Test generation with quantized model"
    )
    
    args = parser.parse_args()
    
    # Initialize converter
    logger.info("Initializing GGUF converter...")
    converter = GGUFConverter(
        llama_cpp_path=args.llama_cpp_path,
        output_dir=args.output_dir
    )
    
    # Convert and quantize
    logger.info(f"Converting model: {args.model_path}")
    logger.info(f"Quantization types: {args.quantization_types}")
    
    quantized_paths = converter.convert_and_quantize(
        model_path=args.model_path,
        quantization_types=args.quantization_types,
        vocab_type=args.vocab_type
    )
    
    logger.info(f"\n✓ Created {len(quantized_paths)} quantized models:")
    for path in quantized_paths:
        logger.info(f"  - {path}")
    
    # Benchmark if requested
    if args.benchmark and quantized_paths:
        logger.info("\nRunning benchmarks...")
        for quant_path in quantized_paths:
            logger.info(f"\nBenchmarking: {quant_path}")
            metrics = converter.benchmark_quantization(
                original_model_path=args.model_path,
                quantized_model_path=quant_path
            )
    
    # Test generation if requested
    if args.test_generation and quantized_paths:
        logger.info("\nTesting generation...")
        test_model = quantized_paths[0]
        logger.info(f"Using model: {test_model}")
        
        try:
            generated = converter.test_quantized_model(
                model_path=test_model,
                test_prompt="Explain machine learning in simple terms.",
                max_tokens=100
            )
            logger.info(f"\nGenerated text:\n{generated}")
        except Exception as e:
            logger.error(f"Generation test failed: {e}")
    
    logger.info("\n✓ Quantization complete!")


if __name__ == "__main__":
    main()
