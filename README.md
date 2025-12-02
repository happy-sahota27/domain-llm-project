# Domain LLM Project

Train & Deploy a Small LLM for Domain-Specific Applications (Healthcare, Legal, Finance)

## 🎯 Project Overview

This project demonstrates end-to-end LLM fine-tuning, evaluation, and deployment with:
- ✅ Domain-specific dataset creation and preprocessing
- ✅ QLoRA (Quantized Low-Rank Adaptation) fine-tuning
- ✅ Comprehensive evaluation with custom metrics (perplexity, ROUGE, BLEU, accuracy)
- ✅ Model quantization to GGUF format
- ✅ FastAPI deployment with REST endpoints
- ✅ Cross-encoder reranker for document retrieval
- ✅ LoRA vs Full Fine-tuning comparison

## 🏗️ Architecture

```
domain-llm-project/
├── src/
│   ├── data/              # Dataset handling
│   ├── training/          # QLoRA training
│   ├── evaluation/        # Metrics & evaluation
│   ├── quantization/      # GGUF conversion
│   ├── api/               # FastAPI deployment
│   └── reranker/          # Document reranking
├── scripts/               # CLI tools
├── configs/               # YAML configurations
├── data/                  # Datasets
├── models/                # Trained models
└── results/               # Evaluation results
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo>
cd domain-llm-project

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your settings
```

### 2. Prepare Dataset

```bash
# Create sample dataset for testing
python scripts/prepare_dataset.py \
    --domain healthcare \
    --sample \
    --num-samples 1000

# Or load from HuggingFace
python scripts/prepare_dataset.py \
    --domain healthcare \
    --dataset-name "medalpaca/medical_meadow_mediqa"
```

### 3. Train Model

```bash
# Train with QLoRA
python scripts/train_model.py \
    --model-name "mistralai/Mistral-7B-v0.1" \
    --dataset-path "data/processed/healthcare_dataset" \
    --output-dir "models/checkpoints" \
    --epochs 3 \
    --batch-size 4
```

### 4. Evaluate Model

```bash
# Run comprehensive evaluation
python scripts/evaluate_model.py \
    --model-path "models/checkpoints" \
    --base-model "mistralai/Mistral-7B-v0.1" \
    --dataset-path "data/processed/healthcare_dataset" \
    --benchmark-speed
```

### 5. Quantize Model

```bash
# Convert to GGUF and quantize
python scripts/quantize_model.py \
    --model-path "models/checkpoints/merged_model" \
    --quantization-types q4_k_m q5_k_m q8_0 \
    --benchmark
```

### 6. Deploy API

```bash
# Deploy with quantized model
python scripts/deploy_api.py \
    --quantized-model "models/quantized/model-q4_k_m.gguf" \
    --host 0.0.0.0 \
    --port 8000

# Or with standard model
python scripts/deploy_api.py \
    --model-path "models/checkpoints" \
    --base-model "mistralai/Mistral-7B-v0.1"
```

Access API docs at: `http://localhost:8000/docs`

## 📊 Features

### Dataset Management
- Multi-domain support (Healthcare, Legal, Finance)
- Automatic preprocessing and validation
- HuggingFace integration
- Custom dataset formats (JSON, CSV)

### Training
- **QLoRA**: 4-bit quantized training with LoRA adapters
- **Full Fine-tuning**: Traditional full parameter training
- **Comparison**: Side-by-side LoRA vs Full Fine-tune
- Memory-efficient with gradient checkpointing
- Distributed training support

### Evaluation
- **Perplexity**: Language model quality
- **Accuracy**: Exact and token-level matching
- **ROUGE**: Text summarization quality
- **BLEU**: Translation/generation quality
- **F1 Score**: Precision-recall balance
- **Speed Benchmarks**: Inference performance

### Quantization
- GGUF format conversion
- Multiple quantization levels (4-bit, 5-bit, 8-bit)
- Size reduction up to 75%
- Minimal accuracy loss

### API
- RESTful endpoints for inference
- Batch generation support
- Document reranking
- CORS enabled
- Auto-generated OpenAPI docs

### Reranker
- Cross-encoder architecture
- Document relevance scoring
- Fine-tunable for domain
- Integrates with API

## 🔧 Configuration

Edit YAML files in `configs/`:

**training_config.yaml**: LoRA parameters, batch size, learning rate
**evaluation_config.yaml**: Metrics, generation settings
**api_config.yaml**: Server settings, endpoints

## 📈 Example Results

```
Training Metrics:
- Training Loss: 0.85
- Validation Loss: 0.92
- Training Time: 2.5 hours
- Trainable Parameters: 42M / 7B (0.6%)

Evaluation Metrics:
- Perplexity: 8.34
- Accuracy: 0.78
- ROUGE-1: 0.65
- BLEU: 0.52
- F1 Score: 0.71

Quantization:
- Original Size: 13.5 GB
- Quantized (Q4_K_M): 3.8 GB (72% reduction)
- Inference Speed: 45 tokens/sec
```

## 🐳 Docker Deployment

```bash
# Build image
docker build -t domain-llm .

# Run container
docker run -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    domain-llm
```

## 🧪 Testing

```bash
# Run test suite
pytest tests/

# Test specific component
pytest tests/test_training.py
pytest tests/test_evaluation.py
```

## 📝 API Usage

```python
import requests

# Generate text
response = requests.post("http://localhost:8000/api/v1/generate", json={
    "prompt": "Explain hypertension",
    "max_tokens": 200,
    "temperature": 0.7
})

print(response.json()["generated_text"])

# Rerank documents
response = requests.post("http://localhost:8000/api/v1/rerank", json={
    "query": "symptoms of diabetes",
    "documents": [
        "Diabetes causes high blood sugar...",
        "Heart disease is...",
        "Common diabetes symptoms include..."
    ],
    "top_k": 2
})

print(response.json()["results"])
```

## 🎓 Training Your Own Domain

1. **Collect Data**: Gather domain-specific text (papers, documents, Q&A)
2. **Preprocess**: Format as instruction-following dataset
3. **Validate**: Ensure data quality
4. **Train**: Use QLoRA for efficiency
5. **Evaluate**: Test on domain-specific metrics
6. **Quantize**: Reduce size for deployment
7. **Deploy**: Serve via FastAPI

## 📚 Technologies Used

- **PyTorch**: Deep learning framework
- **Transformers**: HuggingFace models
- **PEFT**: Parameter-efficient fine-tuning
- **bitsandbytes**: Quantization
- **FastAPI**: API framework
- **sentence-transformers**: Reranking
- **llama.cpp**: GGUF quantization

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional domain support
- More evaluation metrics
- Advanced quantization techniques
- Multi-GPU training optimization
- RAG integration

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- Mistral AI for base models
- HuggingFace for transformers library
- QLoRA paper authors
- llama.cpp team

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Built with ❤️ for domain-specific AI applications**
