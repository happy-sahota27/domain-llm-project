"""
GGUF model converter for quantized inference.
"""

import logging
import subprocess
import os
from pathlib import Path
from typing import Optional, List
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GGUFConverter:
    """Convert models to GGUF format for efficient inference."""
    
    SUPPORTED_QUANTIZATIONS = [
        "q4_0", "q4_1", "q5_0", "q5_1", 
        "q8_0", "q2_k", "q3_k_s", "q3_k_m", 
        "q3_k_l", "q4_k_s", "q4_k_m", "q5_k_s", "q5_k_m", "q6_k"
    ]
    
    def __init__(
        self,
        llama_cpp_path: Optional[str] = None,
        output_dir: str = "models/quantized"
    ):
        """
        Initialize GGUF converter.
        
        Args:
            llama_cpp_path: Path to llama.cpp directory
            output_dir: Directory to save quantized models
        """
        self.llama_cpp_path = llama_cpp_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized GGUFConverter")
    
    def convert_to_gguf(
        self,
        model_path: str,
        output_name: Optional[str] = None,
        vocab_type: str = "spm"  # spm, bpe, or hfft
    ) -> str:
        """
        Convert HuggingFace model to GGUF format.
        
        Args:
            model_path: Path to HuggingFace model
            output_name: Name for output file
            vocab_type: Vocabulary type (spm, bpe, or hfft)
            
        Returns:
            Path to converted GGUF model
        """
        logger.info(f"Converting model to GGUF format: {model_path}")
        
        if output_name is None:
            output_name = f"{Path(model_path).name}.gguf"
        
        output_path = self.output_dir / output_name
        
        # Check if llama.cpp conversion script exists
        if self.llama_cpp_path:
            convert_script = Path(self.llama_cpp_path) / "convert.py"
        else:
            # Try to find convert.py in common locations
            convert_script = "convert.py"
        
        try:
            # Run conversion
            cmd = [
                "python",
                str(convert_script),
                model_path,
                "--outfile", str(output_path),
                "--vocab-type", vocab_type
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("Conversion successful")
            logger.info(result.stdout)
            
            return str(output_path)
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Conversion failed: {e.stderr}")
            raise
        except FileNotFoundError:
            logger.error(
                "convert.py not found. Please install llama.cpp or provide llama_cpp_path"
            )
            raise
    
    def quantize_gguf(
        self,
        gguf_path: str,
        quantization_type: str = "q4_k_m",
        output_name: Optional[str] = None
    ) -> str:
        """
        Quantize GGUF model.
        
        Args:
            gguf_path: Path to GGUF model
            quantization_type: Type of quantization
            output_name: Name for output file
            
        Returns:
            Path to quantized model
        """
        if quantization_type not in self.SUPPORTED_QUANTIZATIONS:
            raise ValueError(
                f"Unsupported quantization type. Choose from: {self.SUPPORTED_QUANTIZATIONS}"
            )
        
        logger.info(f"Quantizing model with {quantization_type}...")
        
        if output_name is None:
            base_name = Path(gguf_path).stem
            output_name = f"{base_name}-{quantization_type}.gguf"
        
        output_path = self.output_dir / output_name
        
        # Check if quantize binary exists
        if self.llama_cpp_path:
            quantize_bin = Path(self.llama_cpp_path) / "quantize"
        else:
            quantize_bin = "quantize"
        
        try:
            # Run quantization
            cmd = [
                str(quantize_bin),
                gguf_path,
                str(output_path),
                quantization_type
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("Quantization successful")
            logger.info(result.stdout)
            
            return str(output_path)
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Quantization failed: {e.stderr}")
            raise
        except FileNotFoundError:
            logger.error(
                "quantize binary not found. Please build llama.cpp or provide llama_cpp_path"
            )
            raise
    
    def convert_and_quantize(
        self,
        model_path: str,
        quantization_types: Optional[List[str]] = None,
        vocab_type: str = "spm"
    ) -> List[str]:
        """
        Convert model to GGUF and create multiple quantized versions.
        
        Args:
            model_path: Path to HuggingFace model
            quantization_types: List of quantization types
            vocab_type: Vocabulary type
            
        Returns:
            List of paths to quantized models
        """
        if quantization_types is None:
            quantization_types = ["q4_k_m", "q5_k_m", "q8_0"]
        
        logger.info(f"Converting and quantizing model: {model_path}")
        
        # First convert to GGUF
        gguf_path = self.convert_to_gguf(model_path, vocab_type=vocab_type)
        
        # Then create quantized versions
        quantized_paths = []
        for quant_type in quantization_types:
            try:
                quant_path = self.quantize_gguf(gguf_path, quant_type)
                quantized_paths.append(quant_path)
            except Exception as e:
                logger.error(f"Failed to create {quant_type} quantization: {e}")
        
        logger.info(f"Created {len(quantized_paths)} quantized versions")
        return quantized_paths
    
    def benchmark_quantization(
        self,
        original_model_path: str,
        quantized_model_path: str
    ) -> dict:
        """
        Compare sizes and basic metrics between original and quantized models.
        
        Args:
            original_model_path: Path to original model
            quantized_model_path: Path to quantized model
            
        Returns:
            Dictionary with comparison metrics
        """
        import os
        
        # Get file sizes
        original_size = self._get_dir_size(original_model_path)
        quantized_size = os.path.getsize(quantized_model_path)
        
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
        size_reduction_percent = (1 - quantized_size / original_size) * 100 if original_size > 0 else 0
        
        metrics = {
            "original_size_mb": original_size / (1024 * 1024),
            "quantized_size_mb": quantized_size / (1024 * 1024),
            "compression_ratio": compression_ratio,
            "size_reduction_percent": size_reduction_percent
        }
        
        logger.info("Quantization Benchmarks:")
        logger.info(f"  Original Size: {metrics['original_size_mb']:.2f} MB")
        logger.info(f"  Quantized Size: {metrics['quantized_size_mb']:.2f} MB")
        logger.info(f"  Compression Ratio: {metrics['compression_ratio']:.2f}x")
        logger.info(f"  Size Reduction: {metrics['size_reduction_percent']:.1f}%")
        
        # Save metrics
        metrics_file = self.output_dir / "quantization_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def _get_dir_size(self, path: str) -> int:
        """Get total size of directory."""
        total = 0
        path = Path(path)
        
        if path.is_file():
            return path.stat().st_size
        
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
        
        return total
    
    def load_quantized_model(self, model_path: str, n_ctx: int = 2048):
        """
        Load quantized GGUF model for inference.
        
        Args:
            model_path: Path to GGUF model
            n_ctx: Context window size
            
        Returns:
            Loaded model (requires llama-cpp-python)
        """
        try:
            from llama_cpp import Llama
            
            logger.info(f"Loading quantized model: {model_path}")
            
            model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=os.cpu_count(),
                n_gpu_layers=-1  # Use GPU if available
            )
            
            logger.info("Model loaded successfully")
            return model
        
        except ImportError:
            logger.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            raise
    
    def test_quantized_model(
        self,
        model_path: str,
        test_prompt: str = "Explain what machine learning is.",
        max_tokens: int = 100
    ) -> str:
        """
        Test quantized model with a sample prompt.
        
        Args:
            model_path: Path to quantized model
            test_prompt: Test prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        model = self.load_quantized_model(model_path)
        
        logger.info(f"Testing with prompt: {test_prompt}")
        
        output = model(
            test_prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            echo=False
        )
        
        generated_text = output['choices'][0]['text']
        logger.info(f"Generated: {generated_text}")
        
        return generated_text
