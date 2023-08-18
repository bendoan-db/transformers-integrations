# Databricks notebook source
# MAGIC %md
# MAGIC # Using 🤗 PEFT & bitsandbytes to finetune a LoRa checkpoint
# MAGIC
# MAGIC
# MAGIC ##### credit: https://www.youtube.com/watch?v=Us5ZFp16PaU&t=481s
# MAGIC
# MAGIC ##### library version:
# MAGIC * accelerate==0.21.0
# MAGIC * transformers==4.31.0
# MAGIC * DBR 12.2LTS ML
# MAGIC * Node type: a2-highgpu-1g[A100], single node
# MAGIC * fsspec[http]>=2021.05.0 (in order to import dataset with this runtime)
# MAGIC * py7zr (in order to work with samsum dataset)
# MAGIC * evaluate (evaluation library)
# MAGIC * rouge_score (evaluate summary metrics with rouge score)

# COMMAND ----------

!pip install --upgrade pip
!pip install -q bitsandbytes datasets accelerate loralib
!pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git
!pip install accelerate==0.21.0
!pip install transformers==4.31.0
!pip install -U "fsspec[http]>=2021.05.0"
!pip install py7zr
!pip install evaluate
!pip install rouge_score

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from huggingface_hub import notebook_login

notebook_login()

# COMMAND ----------

!nvidia-smi -L

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup the model

# COMMAND ----------

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM

print(torch.__version__)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fine-tune and evaluate T5

# COMMAND ----------

model = AutoModelForSeq2SeqLM.from_pretrained('t5-3b')
tokenizer = AutoTokenizer.from_pretrained('t5-3b')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data

# COMMAND ----------

# DBTITLE 1,Prepare Dataset
import transformers
from datasets import load_dataset, concatenate_datasets

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

import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
nltk.download("punkt")

# Metric
metric = evaluate.load("rouge")

# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

# COMMAND ----------

from transformers import DataCollatorForSeq2Seq

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training

# COMMAND ----------

from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datetime import datetime

output_dir = "/dbfs/Users/tian.tan@databricks.com/LLMs/artifacts/t5-3b/outputs_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
dbutils.fs.mkdirs(output_dir)


# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
		per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    fp16=False, # Overflows with fp16
    learning_rate=5e-5,
    num_train_epochs=5,
    # logging & evaluation strategies
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    # metric_for_best_model="overall_f1",
    report_to="tensorboard"
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)


# COMMAND ----------

# Start training
trainer.train()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save and Load adapters from DBFS

# COMMAND ----------

#save to disk
from datetime import datetime
path = "/dbfs/Users/tian.tan@databricks.com/LLMs/artifacts/t5_3b/models_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
dbutils.fs.mkdirs(path)
trainer.model.save_pretrained(path)
tokenizer.save_pretrained(path)


# COMMAND ----------

print(path)

# COMMAND ----------

# Load peft config for pre-trained checkpoint etc. 
peft_model_id = "/dbfs/Users/tian.tan@databricks.com/LLMs/artifacts/t5_3b/models_17-08-2023_21-58-10"
config = PeftConfig.from_pretrained(peft_model_id)

model_download = AutoModelForSeq2SeqLM.from_pretrained(
            config.base_model_name_or_path, 
            load_in_8bit=True, 
            device_map='auto'
            )
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alternatively, share adapters on the 🤗 Hub

# COMMAND ----------

model.push_to_hub("tiantan32/t5-3b-samsum-no-peft",
                  use_auth_token=True,
                  commit_message="basic training",
                  create_pr=1,
                  private=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load adapters from the Hub

# COMMAND ----------

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

peft_model_id = "tiantan32/t5-3b-samsum"
config = PeftConfig.from_pretrained(peft_model_id)
model_download = AutoModelForSeq2SeqLM.from_pretrained(
            config.base_model_name_or_path, 
            load_in_8bit=True, 
            device_map='auto'
            )
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model_download = PeftModel.from_pretrained(model_download, peft_model_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference

# COMMAND ----------

from datasets import load_dataset 
from random import randrange


# Load dataset from the hub and get a sample
dataset = load_dataset("samsum")
sample = dataset['test'][randrange(len(dataset["test"]))]

input_ids = tokenizer(sample["dialogue"], return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = model.generate(input_ids=input_ids, max_new_tokens=10, do_sample=True, top_p=0.9)
print(f"input sentence: {sample['dialogue']}\n{'---'* 20}")

print(f"summary:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}")

# COMMAND ----------

# load test dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
test_dataset = tokenized_dataset["test"].with_format("torch")

# COMMAND ----------

import evaluate
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

# Metric
metric = evaluate.load("rouge")

def evaluate_peft_model(sample,max_target_length=50):
    # generate summary
    outputs = model.generate(input_ids=sample["input_ids"].unsqueeze(0).cuda(), do_sample=True, top_p=0.9, max_new_tokens=max_target_length)    
    prediction = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
    # decode eval sample
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(sample['labels'] != -100, sample['labels'], tokenizer.pad_token_id)
    labels = tokenizer.decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    return prediction, labels


# run predictions
# this can take ~45 minutes
predictions, references = [] , []
for sample in tqdm(test_dataset):
    p,l = evaluate_peft_model(sample)
    predictions.append(p)
    references.append(l)

# compute metric 
rogue = metric.compute(predictions=predictions, references=references, use_stemmer=True)

# print results 
print(f"Rogue1: {rogue['rouge1']* 100:2f}%")
print(f"rouge2: {rogue['rouge2']* 100:2f}%")
print(f"rougeL: {rogue['rougeL']* 100:2f}%")
print(f"rougeLsum: {rogue['rougeLsum']* 100:2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Summary
# MAGIC
# MAGIC 1. Traning a T5 (3B parameters) with PEFT LoRA took 4 hrs on an A100 GPU
# MAGIC 2. According to [this author](https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/peft-flan-t5-int8-summarization.ipynb), the training took ~10:36:00 and cost ~13.22$ for 10h of training a FLAN T5. For comparison a full fine-tuning on FLAN-T5-XXL with the same duration (10h) requires 8x A100 40GBs and costs ~322$.
# MAGIC

# COMMAND ----------


