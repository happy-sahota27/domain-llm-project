"""
QLoRA trainer for efficient fine-tuning of large language models.
"""

import os
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json

from .config import TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QLoRATrainer:
    """Train language models using QLoRA (Quantized Low-Rank Adaptation)."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize QLoRA trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Set up output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized QLoRATrainer with model: {config.model_name}")
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with quantization."""
        logger.info("Loading model and tokenizer...")
        
        # Configure quantization
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=self.config.compute_dtype,
                bnb_4bit_use_double_quant=self.config.use_nested_quant,
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=self.config.compute_dtype
        )
        
        # Prepare model for k-bit training
        if self.config.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        logger.info("Model and tokenizer loaded successfully")
    
    def setup_lora(self):
        """Configure and apply LoRA to the model."""
        logger.info("Setting up LoRA...")
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        logger.info(f"Total parameters: {total_params:,}")
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """
        Tokenize dataset for training.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Tokenized dataset
        """
        logger.info("Tokenizing dataset...")
        
        def tokenize_function(examples):
            # Tokenize texts
            result = self.tokenizer(
                examples[self.config.dataset_text_field],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=False,
                return_tensors=None,
            )
            
            # Create labels (same as input_ids for causal LM)
            result["labels"] = result["input_ids"].copy()
            
            return result
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        logger.info(f"Tokenization complete. Dataset size: {len(tokenized_dataset)}")
        return tokenized_dataset
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            resume_from_checkpoint: Optional checkpoint path to resume from
            
        Returns:
            Training metrics
        """
        # Load model if not already loaded
        if self.model is None:
            self.load_model_and_tokenizer()
            self.setup_lora()
        
        # Tokenize datasets
        train_dataset = self.tokenize_dataset(train_dataset)
        if eval_dataset is not None:
            eval_dataset = self.tokenize_dataset(eval_dataset)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=self.config.save_total_limit,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            optim=self.config.optim,
            lr_scheduler_type=self.config.lr_scheduler_type,
            group_by_length=self.config.group_by_length,
            report_to=self.config.report_to,
            seed=self.config.seed,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train
        logger.info("Starting training...")
        start_time = time.time()
        
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        training_time = time.time() - start_time
        
        # Save model
        self.trainer.save_model()
        
        # Save metrics
        metrics = train_result.metrics
        metrics["training_time"] = training_time
        metrics["training_time_per_epoch"] = training_time / self.config.num_train_epochs
        
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Final training loss: {metrics.get('train_loss', 'N/A')}")
        
        return metrics
    
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            eval_dataset: Evaluation dataset
            
        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Model must be trained or loaded before evaluation")
        
        logger.info("Running evaluation...")
        
        # Tokenize evaluation dataset
        eval_dataset = self.tokenize_dataset(eval_dataset)
        
        # Evaluate
        metrics = self.trainer.evaluate(eval_dataset)
        
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)
        
        logger.info(f"Evaluation complete. Loss: {metrics.get('eval_loss', 'N/A')}")
        
        return metrics
    
    def save_model(self, output_dir: Optional[str] = None):
        """
        Save the trained model.
        
        Args:
            output_dir: Directory to save the model
        """
        if output_dir is None:
            output_dir = self.config.output_dir
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save the model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save config
        config_path = Path(output_dir) / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Model saved to {output_dir}")
    
    def load_trained_model(self, model_path: str):
        """
        Load a previously trained model.
        
        Args:
            model_path: Path to the saved model
        """
        logger.info(f"Loading trained model from {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load base model with quantization
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=self.config.compute_dtype,
                bnb_4bit_use_double_quant=self.config.use_nested_quant,
            )
        else:
            bnb_config = None
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load LoRA weights
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(base_model, model_path)
        
        logger.info("Model loaded successfully")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before generation")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def merge_and_save(self, output_dir: str):
        """
        Merge LoRA weights with base model and save.
        
        Args:
            output_dir: Directory to save merged model
        """
        logger.info("Merging LoRA weights with base model...")
        
        # Merge weights
        merged_model = self.model.merge_and_unload()
        
        # Save
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Merged model saved to {output_dir}")
