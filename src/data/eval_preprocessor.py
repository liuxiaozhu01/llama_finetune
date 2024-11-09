import logging
import numpy as np
import multiprocessing
from functools import partial
from typing import Dict, List, Any
from pytz import timezone
from datetime import datetime
from datasets import Dataset
from transformers import LlamaTokenizer

class EvalDatasetPreprocessor:
    
    def __init__(self, tokenizer: LlamaTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.num_cores = max(multiprocessing.cpu_count() // 3, 1)
        
        self.preprocessors = {
            "winogrande": EvalWinograndePreprocessor(tokenizer, max_length),
        }
        

class EvalWinograndePreprocessor:
    def __init__(self, tokenizer: LlamaTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
