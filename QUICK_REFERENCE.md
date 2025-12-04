# 🚀 Quick Reference - Domain LLM Project

## ⚡ Fast Commands

### Dataset Creation
```bash
# Sample data (testing)
python scripts/prepare_dataset.py --domain healthcare --sample --num-samples 100

# Real data from HuggingFace
python scripts/prepare_dataset.py --domain healthcare \
    --dataset-name "medalpaca/medical_meadow_mediqa" --num-samples 10000

# Custom data
python scripts/prepare_dataset.py --domain healthcare \
    --input-file data/raw/my_data.json
```

### Training
```bash
# Basic training
python scripts/train_model.py \
    --model-name "mistralai/Mistral-7B-v0.1" \
    --dataset-path "data/processed/healthcare_dataset" \
    --epochs 3

# Memory-optimized
python scripts/train_model.py \
    --model-name "mistralai/Mistral-7B-v0.1" \
    --dataset-path "data/processed/healthcare_dataset" \
    --batch-size 2 --gradient-accumulation-steps 8
```

### Evaluation
```bash
# Quick evaluation
python scripts/evaluate_model.py \
    --model-path "models/checkpoints" \
    --dataset-path "data/processed/healthcare_dataset"

# With benchmarks
python scripts/evaluate_model.py \
    --model-path "models/checkpoints" \
    --dataset-path "data/processed/healthcare_dataset" \
    --benchmark-speed
```

### Quantization
```bash
# Recommended quantization
python scripts/quantize_model.py \
    --model-path "models/checkpoints/merged_model" \
    --quantization-types q4_k_m --benchmark
```

### API Deployment
```bash
# Quantized model (recommended)
python scripts/deploy_api.py \
    --quantized-model "models/quantized/model-q4_k_m.gguf"

# Standard model
python scripts/deploy_api.py \
    --model-path "models/checkpoints"
```

### Demo & Testing
```bash
# Quick demo
python quick_demo.py

# Run tests
pytest tests/ -v
```

---

## 📂 Important Paths

| Item | Path |
|------|------|
| Datasets | `data/processed/` |
| Raw Data | `data/raw/` |
| Models | `models/checkpoints/` |
| Quantized | `models/quantized/` |
| Results | `results/evaluation/` |
| Configs | `configs/` |

---

## 🔑 Key Parameters

### Training
- `--epochs`: Number of training epochs (default: 3)
- `--batch-size`: Per-device batch size (default: 4)
- `--learning-rate`: Learning rate (default: 2e-4)
- `--lora-r`: LoRA rank (default: 64)
- `--max-seq-length`: Max sequence length (default: 2048)

### Quantization Types
- `q4_k_m`: ★ Recommended (3.8GB, <3% loss)
- `q5_k_m`: Better quality (4.5GB, <2% loss)
- `q8_0`: Highest quality (7GB, <1% loss)

### API Parameters
- `--host`: Server host (default: 0.0.0.0)
- `--port`: Server port (default: 8000)
- `--workers`: Number of workers (default: 1)

---

## 📊 Status Checks

### Check Dataset
```bash
python -c "from datasets import load_from_disk; \
d = load_from_disk('data/processed/healthcare_dataset'); \
print(f'Train: {len(d[\"train\"])}, Val: {len(d[\"validation\"])}, Test: {len(d[\"test\"])}')"
```

### Check GPU
```bash
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

### API Test
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Generate text
curl -X POST http://localhost:8000/api/v1/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "What is diabetes?", "max_tokens": 100}'
```

---

## 🐛 Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch-size 2

# Increase gradient accumulation
--gradient-accumulation-steps 8

# Reduce sequence length
--max-seq-length 1024
```

### Slow Training
```bash
# Check GPU usage
nvidia-smi -l 1

# Enable gradient checkpointing (default: enabled)
# Reduce logging frequency
--logging-steps 100
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python version
python --version  # Should be 3.9+
```

---

## 📝 File Formats

### Dataset JSON
```json
[
  {
    "instruction": "Explain the medical condition",
    "input": "What is diabetes?",
    "output": "Diabetes is a chronic condition..."
  }
]
```

### CSV Format
```csv
instruction,input,output
"Explain the medical condition","What is diabetes?","Diabetes is..."
```

---

## 🔗 Quick Links

- **Full Documentation**: [README.md](README.md)
- **Run Summary**: [RUN_SUMMARY.md](RUN_SUMMARY.md)
- **API Docs**: http://localhost:8000/docs (when running)
- **GitHub**: [domain-llm-project](https://github.com/happy-sahota27/domain-llm-project)

---

## 💡 Pro Tips

1. **Start Small**: Use `--sample --num-samples 100` for quick testing
2. **Save Checkpoints**: Training saves every 100 steps automatically
3. **Use Quantization**: Deploy with q4_k_m for 72% size reduction
4. **Monitor Training**: Check `logs/` directory for training progress
5. **Test API Locally**: Use `/docs` endpoint for interactive testing

---

## ⏱️ Time Estimates

| Task | Time | Requirements |
|------|------|--------------|
| Dataset Creation | 5 min | CPU only |
| Training | 2-3 hours | 16GB+ GPU |
| Evaluation | 15-30 min | GPU/CPU |
| Quantization | 10-20 min | CPU only |
| API Deployment | 5 min | GPU/CPU |

---

## 🎯 Common Workflows

### Quick Test (No GPU)
```bash
python scripts/prepare_dataset.py --domain healthcare --sample --num-samples 100
python quick_demo.py
```

### Full Pipeline (With GPU)
```bash
# 1. Create dataset
python scripts/prepare_dataset.py --domain healthcare --sample --num-samples 1000

# 2. Train
python scripts/train_model.py \
    --model-name "mistralai/Mistral-7B-v0.1" \
    --dataset-path "data/processed/healthcare_dataset" \
    --epochs 3

# 3. Evaluate
python scripts/evaluate_model.py \
    --model-path "models/checkpoints" \
    --dataset-path "data/processed/healthcare_dataset"

# 4. Quantize
python scripts/quantize_model.py \
    --model-path "models/checkpoints/merged_model" \
    --quantization-types q4_k_m

# 5. Deploy
python scripts/deploy_api.py \
    --quantized-model "models/quantized/model-q4_k_m.gguf"
```

### Production Deployment
```bash
# 1. Use real data
python scripts/prepare_dataset.py \
    --domain healthcare \
    --dataset-name "medalpaca/medical_meadow_mediqa" \
    --num-samples 50000

# 2. Train with more epochs
python scripts/train_model.py \
    --model-name "mistralai/Mistral-7B-v0.1" \
    --dataset-path "data/processed/healthcare_dataset" \
    --epochs 5

# 3. Deploy with Docker
docker build -t domain-llm .
docker run -p 8000:8000 domain-llm
```

---

**Last Updated**: December 4, 2025  
**Version**: 1.0.0  
**Status**: ✅ Ready to use!
