"""
Domain-specific dataset builder for Healthcare, Legal, and Finance domains.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DomainDatasetBuilder:
    """Build and manage domain-specific datasets for LLM fine-tuning."""
    
    SUPPORTED_DOMAINS = ["healthcare", "legal", "finance"]
    
    def __init__(self, domain: str, output_dir: str = "data/processed"):
        """
        Initialize dataset builder.
        
        Args:
            domain: One of 'healthcare', 'legal', or 'finance'
            output_dir: Directory to save processed datasets
        """
        if domain.lower() not in self.SUPPORTED_DOMAINS:
            raise ValueError(f"Domain must be one of {self.SUPPORTED_DOMAINS}")
        
        self.domain = domain.lower()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized DomainDatasetBuilder for {self.domain}")
    
    def load_from_huggingface(
        self,
        dataset_name: str,
        split: Optional[str] = None,
        subset: Optional[str] = None
    ) -> Union[Dataset, DatasetDict]:
        """
        Load dataset from Hugging Face Hub.
        
        Args:
            dataset_name: Name of the dataset on HF Hub
            split: Specific split to load (e.g., 'train', 'test')
            subset: Subset/configuration name
            
        Returns:
            Dataset or DatasetDict
        """
        logger.info(f"Loading dataset: {dataset_name}")
        try:
            dataset = load_dataset(dataset_name, subset, split=split)
            logger.info(f"Successfully loaded {dataset_name}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def load_from_json(self, file_path: str) -> Dataset:
        """
        Load dataset from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Dataset object
        """
        logger.info(f"Loading dataset from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
        
        return Dataset.from_pandas(df)
    
    def load_from_csv(self, file_path: str) -> Dataset:
        """
        Load dataset from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Dataset object
        """
        logger.info(f"Loading dataset from {file_path}")
        df = pd.read_csv(file_path)
        return Dataset.from_pandas(df)
    
    def create_instruction_dataset(
        self,
        data: List[Dict[str, str]],
        instruction_key: str = "instruction",
        input_key: str = "input",
        output_key: str = "output"
    ) -> Dataset:
        """
        Create instruction-tuning format dataset.
        
        Args:
            data: List of dictionaries with instruction, input, output
            instruction_key: Key for instruction field
            input_key: Key for input field
            output_key: Key for output field
            
        Returns:
            Dataset in instruction format
        """
        formatted_data = []
        
        for item in data:
            formatted_item = {
                "instruction": item.get(instruction_key, ""),
                "input": item.get(input_key, ""),
                "output": item.get(output_key, ""),
                "text": self._format_prompt(
                    item.get(instruction_key, ""),
                    item.get(input_key, ""),
                    item.get(output_key, "")
                )
            }
            formatted_data.append(formatted_item)
        
        return Dataset.from_list(formatted_data)
    
    def _format_prompt(self, instruction: str, input_text: str, output: str) -> str:
        """
        Format prompt in Alpaca-style format.
        
        Args:
            instruction: The instruction
            input_text: The input context
            output: The expected output
            
        Returns:
            Formatted prompt string
        """
        if input_text:
            prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        else:
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""
        
        return prompt
    
    def create_qa_dataset(
        self,
        data: List[Dict[str, str]],
        question_key: str = "question",
        answer_key: str = "answer",
        context_key: Optional[str] = None
    ) -> Dataset:
        """
        Create Q&A format dataset.
        
        Args:
            data: List of dictionaries with questions and answers
            question_key: Key for question field
            answer_key: Key for answer field
            context_key: Optional key for context field
            
        Returns:
            Dataset in Q&A format
        """
        formatted_data = []
        
        for item in data:
            context = item.get(context_key, "") if context_key else ""
            formatted_item = {
                "question": item.get(question_key, ""),
                "answer": item.get(answer_key, ""),
                "context": context,
                "text": self._format_qa_prompt(
                    item.get(question_key, ""),
                    item.get(answer_key, ""),
                    context
                )
            }
            formatted_data.append(formatted_item)
        
        return Dataset.from_list(formatted_data)
    
    def _format_qa_prompt(self, question: str, answer: str, context: str = "") -> str:
        """Format Q&A prompt."""
        if context:
            prompt = f"""Context: {context}

Question: {question}

Answer: {answer}"""
        else:
            prompt = f"""Question: {question}

Answer: {answer}"""
        
        return prompt
    
    def split_dataset(
        self,
        dataset: Dataset,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        seed: int = 42
    ) -> DatasetDict:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            dataset: Dataset to split
            train_size: Proportion for training
            val_size: Proportion for validation
            test_size: Proportion for testing
            seed: Random seed
            
        Returns:
            DatasetDict with train, validation, and test splits
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
            "Split sizes must sum to 1.0"
        
        # First split: train and temp (val + test)
        train_test = dataset.train_test_split(
            test_size=(val_size + test_size),
            seed=seed
        )
        
        # Second split: val and test
        val_test = train_test['test'].train_test_split(
            test_size=test_size / (val_size + test_size),
            seed=seed
        )
        
        return DatasetDict({
            'train': train_test['train'],
            'validation': val_test['train'],
            'test': val_test['test']
        })
    
    def save_dataset(self, dataset: Union[Dataset, DatasetDict], name: str):
        """
        Save dataset to disk.
        
        Args:
            dataset: Dataset to save
            name: Name for the saved dataset
        """
        save_path = self.output_dir / name
        dataset.save_to_disk(str(save_path))
        logger.info(f"Dataset saved to {save_path}")
    
    def load_saved_dataset(self, name: str) -> Union[Dataset, DatasetDict]:
        """
        Load previously saved dataset.
        
        Args:
            name: Name of the saved dataset
            
        Returns:
            Loaded dataset
        """
        load_path = self.output_dir / name
        dataset = Dataset.load_from_disk(str(load_path))
        logger.info(f"Dataset loaded from {load_path}")
        return dataset
    
    def get_domain_specific_datasets(self) -> Dict[str, str]:
        """
        Get recommended datasets for each domain.
        
        Returns:
            Dictionary of domain-specific dataset recommendations
        """
        recommendations = {
            "healthcare": [
                "medalpaca/medical_meadow_medical_flashcards",
                "medalpaca/medical_meadow_mediqa",
                "bigbio/pubmed_qa",
                "BI55/MedText"
            ],
            "legal": [
                "pile-of-law/pile-of-law",
                "lexlms/lex_glue",
                "nguha/legalbench",
                "law_stack_exchange"
            ],
            "finance": [
                "gbharti/finance-alpaca",
                "FinGPT/fingpt-sentiment-train",
                "financial_phrasebank",
                "zeroshot/twitter-financial-news-sentiment"
            ]
        }
        
        return recommendations.get(self.domain, [])
    
    def create_sample_dataset(self, num_samples: int = 100) -> Dataset:
        """
        Create a small sample dataset for testing.
        
        Args:
            num_samples: Number of samples to create
            
        Returns:
            Sample dataset
        """
        samples = []
        
        if self.domain == "healthcare":
            for i in range(num_samples):
                samples.append({
                    "instruction": "Explain the medical condition",
                    "input": f"What is hypertension?",
                    "output": f"Hypertension, or high blood pressure, is a condition where the force of blood against artery walls is too high."
                })
        elif self.domain == "legal":
            for i in range(num_samples):
                samples.append({
                    "instruction": "Explain the legal term",
                    "input": f"What is tort law?",
                    "output": f"Tort law is a branch of civil law that addresses breaches of civil duties owed to others."
                })
        elif self.domain == "finance":
            for i in range(num_samples):
                samples.append({
                    "instruction": "Explain the financial concept",
                    "input": f"What is compound interest?",
                    "output": f"Compound interest is interest calculated on the initial principal and accumulated interest from previous periods."
                })
        
        return self.create_instruction_dataset(samples)
