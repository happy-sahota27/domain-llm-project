"""
Data preprocessing utilities for cleaning and preparing text data.
"""

import re
import logging
from typing import List, Dict, Optional, Callable
from datasets import Dataset
import html

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess text data for LLM training."""
    
    def __init__(self, max_length: int = 2048, min_length: int = 10):
        """
        Initialize preprocessor.
        
        Args:
            max_length: Maximum sequence length
            min_length: Minimum sequence length
        """
        self.max_length = max_length
        self.min_length = min_length
        logger.info(f"Initialized DataPreprocessor (max_length={max_length}, min_length={min_length})")
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing unwanted characters and formatting.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def remove_special_characters(self, text: str, keep_punctuation: bool = True) -> str:
        """
        Remove special characters from text.
        
        Args:
            text: Input text
            keep_punctuation: Whether to keep punctuation marks
            
        Returns:
            Cleaned text
        """
        if keep_punctuation:
            # Keep alphanumeric, spaces, and common punctuation
            pattern = r'[^a-zA-Z0-9\s.,!?;:\-\'"()\[\]{}]'
        else:
            # Keep only alphanumeric and spaces
            pattern = r'[^a-zA-Z0-9\s]'
        
        text = re.sub(pattern, '', text)
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with single newline
        text = re.sub(r'\n+', '\n', text)
        
        # Remove spaces before punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        return text.strip()
    
    def filter_by_length(self, dataset: Dataset, text_column: str = "text") -> Dataset:
        """
        Filter dataset by text length.
        
        Args:
            dataset: Input dataset
            text_column: Name of the text column
            
        Returns:
            Filtered dataset
        """
        def length_filter(example):
            text_len = len(example[text_column].split())
            return self.min_length <= text_len <= self.max_length
        
        original_size = len(dataset)
        filtered_dataset = dataset.filter(length_filter)
        filtered_size = len(filtered_dataset)
        
        logger.info(f"Filtered dataset from {original_size} to {filtered_size} examples")
        return filtered_dataset
    
    def remove_duplicates(self, dataset: Dataset, column: str = "text") -> Dataset:
        """
        Remove duplicate entries from dataset.
        
        Args:
            dataset: Input dataset
            column: Column to check for duplicates
            
        Returns:
            Dataset without duplicates
        """
        original_size = len(dataset)
        
        # Convert to pandas, drop duplicates, convert back
        df = dataset.to_pandas()
        df = df.drop_duplicates(subset=[column], keep='first')
        deduplicated_dataset = Dataset.from_pandas(df, preserve_index=False)
        
        deduplicated_size = len(deduplicated_dataset)
        logger.info(f"Removed {original_size - deduplicated_size} duplicates")
        
        return deduplicated_dataset
    
    def apply_preprocessing(
        self,
        dataset: Dataset,
        columns: List[str],
        custom_fn: Optional[Callable] = None
    ) -> Dataset:
        """
        Apply preprocessing to specified columns.
        
        Args:
            dataset: Input dataset
            columns: List of column names to preprocess
            custom_fn: Optional custom preprocessing function
            
        Returns:
            Preprocessed dataset
        """
        def preprocess_function(examples):
            for col in columns:
                if col in examples:
                    if isinstance(examples[col], list):
                        examples[col] = [
                            self.clean_text(text) for text in examples[col]
                        ]
                        if custom_fn:
                            examples[col] = [
                                custom_fn(text) for text in examples[col]
                            ]
                    else:
                        examples[col] = self.clean_text(examples[col])
                        if custom_fn:
                            examples[col] = custom_fn(examples[col])
            return examples
        
        logger.info(f"Applying preprocessing to columns: {columns}")
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            desc="Preprocessing"
        )
        
        return processed_dataset
    
    def remove_empty_examples(self, dataset: Dataset, columns: List[str]) -> Dataset:
        """
        Remove examples where specified columns are empty.
        
        Args:
            dataset: Input dataset
            columns: Columns to check for empty values
            
        Returns:
            Dataset without empty examples
        """
        def not_empty(example):
            for col in columns:
                if col in example:
                    if not example[col] or (isinstance(example[col], str) and not example[col].strip()):
                        return False
            return True
        
        original_size = len(dataset)
        filtered_dataset = dataset.filter(not_empty)
        filtered_size = len(filtered_dataset)
        
        logger.info(f"Removed {original_size - filtered_size} empty examples")
        return filtered_dataset
    
    def balance_dataset(
        self,
        dataset: Dataset,
        label_column: str,
        max_samples_per_class: Optional[int] = None
    ) -> Dataset:
        """
        Balance dataset by downsampling majority classes.
        
        Args:
            dataset: Input dataset
            label_column: Column containing class labels
            max_samples_per_class: Maximum samples per class
            
        Returns:
            Balanced dataset
        """
        df = dataset.to_pandas()
        
        # Count samples per class
        class_counts = df[label_column].value_counts()
        logger.info(f"Class distribution before balancing:\n{class_counts}")
        
        # Determine target count
        if max_samples_per_class is None:
            max_samples_per_class = class_counts.min()
        
        # Sample from each class
        balanced_dfs = []
        for label in class_counts.index:
            class_df = df[df[label_column] == label]
            if len(class_df) > max_samples_per_class:
                class_df = class_df.sample(n=max_samples_per_class, random_state=42)
            balanced_dfs.append(class_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        balanced_dataset = Dataset.from_pandas(balanced_df, preserve_index=False)
        
        logger.info(f"Balanced dataset size: {len(balanced_dataset)}")
        return balanced_dataset
    
    def add_special_tokens(
        self,
        dataset: Dataset,
        text_column: str = "text",
        bos_token: str = "<s>",
        eos_token: str = "</s>"
    ) -> Dataset:
        """
        Add special tokens to text.
        
        Args:
            dataset: Input dataset
            text_column: Column containing text
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
            
        Returns:
            Dataset with special tokens added
        """
        def add_tokens(example):
            example[text_column] = f"{bos_token}{example[text_column]}{eos_token}"
            return example
        
        return dataset.map(add_tokens)
    
    def create_chat_format(
        self,
        dataset: Dataset,
        user_column: str = "input",
        assistant_column: str = "output",
        system_prompt: Optional[str] = None
    ) -> Dataset:
        """
        Convert dataset to chat format.
        
        Args:
            dataset: Input dataset
            user_column: Column containing user messages
            assistant_column: Column containing assistant responses
            system_prompt: Optional system prompt
            
        Returns:
            Dataset in chat format
        """
        def format_chat(example):
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": example[user_column]})
            messages.append({"role": "assistant", "content": example[assistant_column]})
            
            example["messages"] = messages
            return example
        
        return dataset.map(format_chat)
    
    def truncate_text(
        self,
        dataset: Dataset,
        column: str = "text",
        max_length: Optional[int] = None
    ) -> Dataset:
        """
        Truncate text to maximum length.
        
        Args:
            dataset: Input dataset
            column: Column to truncate
            max_length: Maximum length (uses self.max_length if None)
            
        Returns:
            Dataset with truncated text
        """
        if max_length is None:
            max_length = self.max_length
        
        def truncate(example):
            words = example[column].split()
            if len(words) > max_length:
                example[column] = ' '.join(words[:max_length])
            return example
        
        return dataset.map(truncate)
    
    def compute_statistics(self, dataset: Dataset, text_column: str = "text") -> Dict:
        """
        Compute statistics about the dataset.
        
        Args:
            dataset: Input dataset
            text_column: Column containing text
            
        Returns:
            Dictionary of statistics
        """
        lengths = []
        char_counts = []
        
        for example in dataset:
            text = example[text_column]
            words = text.split()
            lengths.append(len(words))
            char_counts.append(len(text))
        
        import numpy as np
        
        stats = {
            "num_examples": len(dataset),
            "avg_word_length": np.mean(lengths),
            "median_word_length": np.median(lengths),
            "min_word_length": np.min(lengths),
            "max_word_length": np.max(lengths),
            "avg_char_length": np.mean(char_counts),
            "total_words": sum(lengths),
            "total_chars": sum(char_counts)
        }
        
        logger.info(f"Dataset statistics: {stats}")
        return stats
