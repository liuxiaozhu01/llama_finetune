import re
import logging
import copy
import pandas as pd
import multiprocessing
from typing import Dict, List, Any, Sequence
from datetime import datetime
from datasets import Dataset, concatenate_datasets
from transformers import LlamaTokenizer, PreTrainedTokenizer
from tqdm import tqdm
import torch

from .special_token import IGNORE_INDEX

def _tokenize_fn(strings: Sequence[str], tokenizer: PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

class TrainDatasetTonkenizePreprocessor:
    def __init__(self, tokenizer: LlamaTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
        self.num_cores = max(multiprocessing.cpu_count() // 3, 1)
        
        self.preprocessors = {
            "alpaca": AlpacaPreprocessor(tokenizer, max_length),
            "winogrande": WinograndePreprocessor(tokenizer, max_length),
        }
        
    def __call__(self, datasets: Dict[str, Dataset]) -> Dataset :
        dataset_names = list(datasets.keys())
        
        preprocessed_datasets = []
        for dataset_name in dataset_names:
            dataset = datasets[dataset_name]
            
            if dataset_name in self.preprocessors:
                logging.info(f"Preprocessing and Tokenize Dataset | {dataset_name}")
                preprocessor = self.preprocessors[dataset_name]
                
                # Preprocessing and encoding dataset
                preprocess_fn = preprocessor.preprocess
                preprocessed = dataset.map(preprocess_fn, batched=True, num_proc=self.num_cores, remove_columns=dataset.column_names)

                # Set the format of the dataset to torch.Tensor
                preprocessed.set_format(type='torch', columns=preprocessed.column_names)
                                
                ## For logging ##
                # Count the data which length is longer then sequence_max_length
                data_longer_then_sequence_max_length = 0
                for d in preprocessed :
                    if len(d["input_ids"]) > self.max_length :
                        data_longer_then_sequence_max_length += 1
                logging.info(f"### The number of data which length is longer then sequence_max_length\n{data_longer_then_sequence_max_length}\n")
                
                # Logging preprocessed dataset's input_id and label example
                preprocessed_input_id = preprocessed[0]["input_ids"]
                preprocessed_input_text = self.tokenizer.decode(preprocessed_input_id)

                preprocessed_label = preprocessed[0]["labels"]
                preprocessed_label = [l for l in preprocessed_label if l >= 0]
                preprocessed_label_text = self.tokenizer.decode(preprocessed_label)

                logging.info(f"Preprocessed dataset | {dataset_name}\n### Input\n{preprocessed_input_text}\n### Label\n{preprocessed_label_text}\n\n")
                #################
                
                preprocessed_datasets.append(preprocessed)
        
        preprocessed_datasets = concatenate_datasets(preprocessed_datasets)
        return preprocessed_datasets
    
class AlpacaPreprocessor:
    def __init__(self, tokenizer: LlamaTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.PROMPT_DICT = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:"
            ),
        }
        
    def preprocess(self, examples: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        prompt_input, prompt_no_input = self.PROMPT_DICT["prompt_input"], self.PROMPT_DICT["prompt_no_input"]
        
        instructions = examples["instruction"]
        input_texts = examples["input"]
        output_texts = examples["output"]
        
        sources = [
            prompt_input.format(instruction=instruction, input=input_text) if input_text != "" else prompt_no_input.format(instruction=instruction)
            for instruction, input_text in zip(instructions, input_texts)
        ]
        targets = [f"{output_text}{self.tokenizer.eos_token}" for output_text in output_texts]
        
        # sources = [
        #     prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        #     for example in examples
        # ]
        # targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in examples]
        
        examples_combined = [s + t for s, t in zip(sources, targets)]
        examples_tokenized = _tokenize_fn(examples_combined, self.tokenizer)
        sources_tokenized = _tokenize_fn(sources, self.tokenizer)
        
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
        
        return dict(input_ids=input_ids, labels=labels)
        
class WinograndePreprocessor:
    def __init__(self, tokenizer: LlamaTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def preprocess(self, examples: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        sentences = examples["sentence"]
        option1s = examples["option1"]
        option2s = examples["option2"]
        answers = examples["answer"]
        
        sources, targets = [], []
        for sentence, option1, option2, answer in zip(sentences, option1s, option2s, answers):
            answer_text = option1 if answer == 1 else option2

            option_idx = sentence.index("_")
            """ 1. same as 'sangHa0411/Llama-Instruction-Tuning' """
            # source_text = sentence[:option_idx]
            # target_text = answer_text + sentence[option_idx+1:]
            """ 2. more similar to lm_eval """
            source_text = sentence[:option_idx] + answer_text
            target_text = sentence[option_idx+1:]
            
            sources.append(f"Sentence: {source_text}")
            targets.append(target_text)     # is eos_token needed?
            
        examples_combined = [s + t for s, t in zip(sources, targets)]
        examples_tokenized = _tokenize_fn(examples_combined, self.tokenizer)
        sources_tokenized = _tokenize_fn(sources, self.tokenizer)
        
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
        
        return dict(input_ids=input_ids, labels=labels)
        