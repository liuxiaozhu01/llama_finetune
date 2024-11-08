from typing import List, Optional
import transformers
import logging

from .collator import DataCollatorForSupervisedDataset
from .dataset_loader import TrainDatasetLoader
from .preprocessor import TrainDatasetTonkenizePreprocessor
from .special_token import smart_tokenizer_and_embedding_resize


def get_train_dataset(
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    max_length: int,
    datasets: str,
    dataset_sizes: str,
    cache_dir: Optional[str]=None,
    random_seed: int=42,
):
    # Get training datasets
    logging.info(f"Load train dataset: {datasets}")
    dataset_loader = TrainDatasetLoader(random_seed, datasets, dataset_sizes, cache_dir)
    train_dataset = dataset_loader.load()
    
    # add special tokens
    # logging.info("Add special tokens and resize embedding")
    # smart_tokenizer_and_embedding_resize(tokenizer, model)
    
    # preprocess and tokenize the dataset
    logging.info("Preprocess and tokenize the dataset")
    train_dataset_preprocessor = TrainDatasetTonkenizePreprocessor(tokenizer=tokenizer, max_length=max_length)
    tokenized_train_dataset = train_dataset_preprocessor(train_dataset)
    
    # dataset collator
    logging.info("Create dataset collator")
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    return dict(train_dataset=tokenized_train_dataset, data_collator=data_collator)