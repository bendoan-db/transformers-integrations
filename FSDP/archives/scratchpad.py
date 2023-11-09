# Databricks notebook source
# MAGIC %pip install py7zr

# COMMAND ----------

import transformers
from datasets import load_dataset, concatenate_datasets
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM

dataset = load_dataset('samsum')

# The maximum total input sequence length after tokenization. 
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["dialogue"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization. 
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["summary"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
print(f"Max target length: {max_target_length}")

# COMMAND ----------

def preprocess_function(sample,padding="max_length"):
    # add prefix to the input for t5
    inputs = ["summarize: " + item for item in sample["dialogue"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

# COMMAND ----------

tokenized_inputs

# COMMAND ----------

tokenized_dataset

# COMMAND ----------

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
bert_datasets = load_dataset("glue", "mrpc")

def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs
      
bert_tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", "sentence1", "sentence2"],
        )

# COMMAND ----------

bert_tokenized_datasets

# COMMAND ----------

tokenized_dataset

# COMMAND ----------

tokenized_dataset

# COMMAND ----------

 import torch 
 print("Cuda support:", torch.cuda.is_available(),":", torch.cuda.device_count(), "devices")

# COMMAND ----------

print(torch.cuda.is_available())

# COMMAND ----------

import accelerate
accelerate.__version__

# COMMAND ----------

import torch
torch.__version__

# COMMAND ----------

import mlflow
mlflow.__version__

# COMMAND ----------

|
