# Databricks notebook source
# MAGIC %md
# MAGIC # Experiment Results
# MAGIC
# MAGIC Below we track the high level experiment results for various evaluations performed with different models, training frameworks, and parameters

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Experiment: Cost Performance of FSDP, PEFT, and Multi-GPU finetuning
# MAGIC
# MAGIC In this experiment, we evaluate the cost performance of training various T5-3B using single GPU and multi - finetuning, with both PEFT and full weight finetuning
# MAGIC
# MAGIC - Model: t5-3B
# MAGIC - Dataset: Samsum chat dataset
# MAGIC
# MAGIC ##### Training Time and Inference
# MAGIC
# MAGIC |       | **Full Weight** | **PEFT**     |
# MAGIC | :---        |    :----:   |          ---: |
# MAGIC | **Single-GPU**      | TBD       | Training: ~4 hours // Batch Inference (800 records): ~2.65 hours   |
# MAGIC | **Multi-GPU**   | 10+ hours then OOM       | TBD      |
# MAGIC
# MAGIC
# MAGIC
# MAGIC ##### Approximate Cost Per Run (Compute + Databricks Licensing)
# MAGIC
# MAGIC |       | **Full Weight** | **PEFT**     |
# MAGIC | :---        |    :----:   |          ---: |
# MAGIC | **Single-GPU**      | TBD       | Training:    |
# MAGIC | **Multi-GPU**   | TBD        | TBD      |
# MAGIC
# MAGIC
# MAGIC
# MAGIC ##### Prompt Evaluation

# COMMAND ----------

# DBTITLE 0,Install Libraries
!pip install --upgrade pip
!pip install -q bitsandbytes datasets accelerate loralib
!pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git
!pip install accelerate==0.21.0
!pip install transformers==4.31.0
!pip install -U "fsspec[http]>=2021.05.0"
!pip install py7zr

# COMMAND ----------

# DBTITLE 1,Load model from the Hub
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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Learnings so far
# MAGIC - A10s are not suitable for LLM training, even with lots of memory saving optimizations
# MAGIC - The LLM training space is a **mess** in terms of documentation. 
# MAGIC     - There is a severe lack of intuitive documentation that explains how these models work and what parameters you can poke/play with to optimize your models
# MAGIC - `accelerate` provides a fantastic framework for quickly standing up distributed training, specifically Fully Sharded Data Parrellel `FSDP` and `DeepSpeed`
# MAGIC - `PEFT` has potential in terms of memory savings, but at signficant performance cost. More testing is required to evaluate the optimal usage
# MAGIC - If you have an LLM opportunity, **engage an expert**, it's a very dense and there's not a ton of centralized documentation or standardization
# MAGIC
# MAGIC ### Next Steps
# MAGIC - Test `FSDP` on an A100 cluster
# MAGIC - Test on larger, more effective models (`Llamav2`, `Platypus12B`, `MPT`)
# MAGIC - Integrate with MLflow for better model tracking

# COMMAND ----------


