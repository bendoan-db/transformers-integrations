# Databricks notebook source
from pynvml import *
import mlflow
import argparse
import logging
from logging import log
import py7zr
import evaluate
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed, AutoModelForSeq2SeqLM, LlamaForCausalLM, LlamaTokenizer
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset
from llama_recipes.configs.datasets import samsum_dataset

from accelerate import Accelerator, DistributedType
from decimal import *
from datetime import datetime
import time
import pynvml

#check to make sure gpu mem isn't being used by any other processes
def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

# COMMAND ----------

TOKEN="TOKEN_HERE"
model_name="meta-llama/Llama-2-7b-hf"
model = LlamaForCausalLM.from_pretrained(model_name, token=TOKEN, torch_dtype=torch.float16).to("cuda")

# COMMAND ----------

#model.to('cpu')
print_gpu_utilization()

# COMMAND ----------

tokenizer = LlamaTokenizer.from_pretrained(model_name, token=TOKEN)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
def tokenize_function(examples):
    return tokenizer(examples["input_ids"], padding="max_length", truncation=True, max_length=256)
      

dataset = get_preprocessed_dataset(tokenizer, samsum_dataset).train_test_split(test_size=0.3)

# COMMAND ----------

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# COMMAND ----------

len(dataset["train"][100]["input_ids"])

# COMMAND ----------

dataset["train"][0]

# COMMAND ----------

len(dataset["train"][0]["input_ids"])

# COMMAND ----------

def truncate_function(examples, max_length=1024):
    # Truncate input_ids to max_length
    examples["input_ids"] = examples["input_ids"][:max_length]

    # Truncate input_ids to max_length
    examples["labels"] = examples["labels"][:max_length]
    
    # Also truncate attention_mask if it exists in your dataset
    if "attention_mask" in examples:
        examples["attention_mask"] = examples["attention_mask"][:max_length]
    
    return examples


tokenized_dataset = dataset.map(
  truncate_function,
  batched=True
)

# COMMAND ----------

len(tokenized_dataset["train"][0]["input_ids"])

# COMMAND ----------


