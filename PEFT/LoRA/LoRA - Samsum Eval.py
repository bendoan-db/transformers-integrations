# Databricks notebook source
!pip install --upgrade pip
!pip install -q bitsandbytes datasets accelerate loralib
!pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git
!pip install accelerate==0.21.0
!pip install transformers==4.31.0
!pip install -U "fsspec[http]>=2021.05.0"
!pip install py7zr

# COMMAND ----------

# DBTITLE 1,Load Model
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

peft_model_id = "tiantan32/t5-3b-samsum"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(
            config.base_model_name_or_path, 
            load_in_8bit=True, 
            device_map='auto'
            )
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

# COMMAND ----------

# DBTITLE 1,Prompt Evaluation Example
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
