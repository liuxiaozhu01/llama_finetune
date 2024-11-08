import os
import torch
import random
import logging
import numpy as np
from pytz import timezone
from typing import Tuple
from datetime import datetime

from datasets import disable_caching
from transformers import (
    LlamaConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    EarlyStoppingCallback, Trainer, TrainingArguments
)
import transformers

from peft import LoraConfig, TaskType, get_peft_model

from data import get_train_dataset
from data.special_token import smart_tokenizer_and_embedding_resize

def get_optimizer_and_scheduler(model, train_dataset, config):
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )

    kwargs_lr_scheduler = {
        "optimizer": optimizer,
        "num_warmup_steps": config.num_warmup_steps,
        "num_training_steps": (
            (len(train_dataset) - 1) // (config.finetune_train_batch_size * config.gradient_accumulation_steps) + 1
        )
        * config.epochs,
    }
    if config.lr_scheduler_type in ("cosine", "cosine_with_warmup"):
        lr_scheduler = transformers.get_cosine_schedule_with_warmup(**kwargs_lr_scheduler)
    elif config.lr_scheduler_type in ("linear", "linear_with_warmup"):
        lr_scheduler = transformers.get_linear_schedule_with_warmup(**kwargs_lr_scheduler)
    else:
        raise NotImplementedError

    return optimizer, lr_scheduler

def get_model_and_tokenizer(args) -> Tuple[LlamaConfig, LlamaForCausalLM, LlamaTokenizer]:

    logging.info(f"model | {args.model_path} | tokenizer: {args.tokenizer_path}")
    model_path = args.model_path

    # Load config
    config = LlamaConfig.from_pretrained(model_path)

    if args.gradient_checkpointing :
        config.gradient_checkpointing = True

    config.embd_pdrop = args.dropout_rate
    config.resid_pdrop = args.dropout_rate
    config.attn_pdrop = args.dropout_rate
    
    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)

    # Load huggingface model
    logging.info("Load huggingface model")
    hf_model = LlamaForCausalLM.from_pretrained(model_path)
    
    # add special tokens
    logging.info("Add special tokens and resize embedding")
    smart_tokenizer_and_embedding_resize(tokenizer, hf_model)

    return config, hf_model, tokenizer

def train(args):

    # Loading config, model and tokenizer
    config, hf_model, tokenizer = get_model_and_tokenizer(args)

    # Setting random seed
    seed_everything(args.random_seed)
    
    # disable_caching()
    
    train_dataset_module = get_train_dataset(tokenizer=tokenizer, model=hf_model, max_length=args.max_sequence_length, 
                                             datasets=args.train_datasets, dataset_sizes=args.dataset_sizes, cache_dir=args.cache_dir,
                                             random_seed=args.random_seed)
    
    train_dataset, data_collator = train_dataset_module['train_dataset'], train_dataset_module['data_collator']
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=args.lora_target_modules.split(","),
    )
    
    lora_model = get_peft_model(hf_model, lora_config)
    lora_model.print_trainable_parameters()
    
    torch.cuda.empty_cache()
    # create optimizer and scheduler
    optimizer, lr_scheduler = get_optimizer_and_scheduler(lora_model, train_dataset, args)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,  # output directory
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.per_device_eval_batch_size,  # batch size for evaluation
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        disable_tqdm=False,
        load_best_model_at_end=True,
        eval_steps=args.eval_steps,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.evaluation_strategy,      # args.evaluation_strategy if finetune_test_loader is not None else "no",
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # lower eval_loss is better,
        gradient_checkpointing=True,
    )
    
    trainer = Trainer(
        args=training_args,
        model=lora_model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=None,  # not implement yet
        optimizers=(optimizer, lr_scheduler),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience,
                                            early_stopping_threshold=args.early_stopping_delta)]
    )

    # required to enable gradient_checkpointing
    lora_model.enable_input_require_grads()

    lora_model.train()
    trainer.train()
    
    # save the final checkpoint
    trainer.save_model(os.path.join(args.output_dir, "checkpoint-final"))
    

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="llama_finetuning")

    # Dataset names
    parser.add_argument("--train_datasets", type=str, default="[alpaca,cot-collection]", help="instruction datasets | possible datasets [alpaca, cot-collections, slimorca, openorca-multiplechoice, arc, mmlu, gsm8k, winogrande]")
    parser.add_argument("--dataset_sizes", type=str, default="[1.0,10%]", help="instruction dataset ratios")
    parser.add_argument("--evaluation_datasets", type=str, default="[ai2_arc,Rowan/hellaswag]", help="evaluation datasets | possible datasets [arc, hellaswag, gsm8k, truthful_qa-generation, truthful_qa-multiple_choice, winogrande]")
    parser.add_argument("--evaluation_shots", type=str, default="[0,0]", help="shot size for evaluation")

    # Model Name
    parser.add_argument("--model_name", type=str, default="llama-2-7b-hf", help="model's name")
    parser.add_argument("--run_name", type=str, default=None, help="A descriptor for the run. used for wandb logging")

    # Random Seed
    parser.add_argument("--random_seed", type=int, default=42, help="fix random seed in torch, random, numpy")

    # Gradient checkpointing
    parser.add_argument("--gradient_checkpointing", type=bool, default=False, help="use gradient checkpointing for training")

    # Sequence Length and Generation Length
    parser.add_argument("--dropout_rate", type=float, default=0.0, help="dropout rate for llm training")

    # Sequence Length and Generation Length
    parser.add_argument("--max_sequence_length", type=int, default=512, help="llm model max sequence length for training")
    parser.add_argument("--eval_sequence_max_length", type=int, default=1536, help="llm model max sequence length for evaluation")
    parser.add_argument("--generation_max_length", type=int, default=1024, help="generation max length")

    # Data & Logging Path
    parser.add_argument("--logging_dir", type=str, default="/project/llama-instruction-tuning/exps/logging", help="path for evaluation prediction results")
    parser.add_argument("--output_dir", type=str, default="/mnt/disks-standard/persist/t5/llama-alpaca/exps/checkpoints", help="model checkpoint path")
    parser.add_argument("--cache_dir", type=str, default="/mnt/disks-standard/persist/huggingface", help="dataset cache path")

    # Model evaluation & save strategy
    parser.add_argument("--evaluation_strategy", type=str, default="none", help="do model evaluation during training | possible strategies [epoch, steps]")
    parser.add_argument("--eval_steps", type=int, default=1000, help="every this size training step, do model evaluation")
    parser.add_argument("--save_strategy", type=str, default="none", help="do model save during training | possible strategies [epoch, steps]")
    parser.add_argument("--save_steps", type=int, default=1000, help="every this size training step, do model save")
    parser.add_argument("--save_total_limit", type=int, default=20)
    

    # Model & Tokenizer path
    parser.add_argument("--tokenizer_path", type=str, default="/mnt/disks-standard/persist/llama/llama-2-7b-hf", help="path for evaluation prediction results")
    parser.add_argument("--model_path", type=str, default="/mnt/disks-standard/persist/llama/llama-2-7b-hf", help="path for evaluation prediction results")

    # Tokenizer padding side
    parser.add_argument("--padding_side", type=str, default="left", help="tokenizer padding side | possible sides [left, right]")

    # Epoch & Batch size
    parser.add_argument("--num_train_epochs", type=int, default=3, help="num_train_epochs for training")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="training batch size")
    # parser.add_argument("--per_device_eval_forward_batch_size", type=int, default=4, help="evaluation batch size")
    # parser.add_argument("--per_device_eval_generate_batch_size", type=int, default=4, help="evaluation batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="evaluation batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="gradient accumulation steps")
    
    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="dataset")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="dataset")

    # Scheduler
    parser.add_argument("--lr_scheduler_type", type=str, default="constant", help="type of learning rate scheduler")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="warmup ratio of linear learning rate scheduler")
    
    # LoRA
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument('--lora_target_modules', type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj", help='lora target modules')
    parser.add_argument('--lora_bias', type=str, default="none")

    # Logging
    parser.add_argument("--logging_steps", type=int, default=100, help="Number of update steps between two logs if logging_strategy is 'step")

    args = parser.parse_args()

    args.run_name = f"MODEL_NAME:{args.model_name}-DATASETS:{args.instruction_datasets}-DATASET_SIZES:{args.dataset_sizes}-EP:{args.num_train_epochs}-LR:{args.learning_rate}-BS:{args.per_device_train_batch_size}-WR:{args.warmup_ratio}-WD:{args.weight_decay}"
    args.output_dir = f"{args.output_dir}/{args.run_name}"

    logging.info(f"Training arguments: {args}")
    train(args)


