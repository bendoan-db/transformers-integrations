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
# MAGIC * Node type: a2-highgpu-1g[A100]

# COMMAND ----------

!pip install --upgrade pip

# COMMAND ----------

!pip install -q bitsandbytes datasets accelerate loralib
!pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git

# COMMAND ----------

# DBTITLE 1,Run
!pip install accelerate==0.21.0
!pip install transformers==4.31.0

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
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

print(torch.__version__)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setting up the LoRa Adapters

# COMMAND ----------

model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-7b1",
    load_in_8bit=True,
    device_map='auto',
)

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Freezing the original weights
# MAGIC

# COMMAND ----------

for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

# COMMAND ----------

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# COMMAND ----------

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16, #attention heads
    lora_alpha=32, #alpha scaling
    # target_modules=["q_proj", "v_proj"], #if you know the
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data

# COMMAND ----------

import transformers
from datasets import load_dataset
data = load_dataset("Abirate/english_quotes")


# COMMAND ----------

def merge_columns(example):
    example["prediction"] = example["quote"] + " ->: " + str(example["tags"])
    return example

data['train'] = data['train'].map(merge_columns)
data['train']["prediction"][:5]

# COMMAND ----------

data['train'][0]

# COMMAND ----------

data = data.map(lambda samples: tokenizer(samples['prediction']), batched=True)

# COMMAND ----------

data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training

# COMMAND ----------

from datetime import datetime
path = "/dbfs/Users/tian.tan@databricks.com/LLMs/artifacts/lora-bloom7b/outputs_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
dbutils.fs.mkdirs(path)
print(path)

# COMMAND ----------

trainer = transformers.Trainer(
    model=model,
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=200,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir=path
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Share adapters on the 🤗 Hub

# COMMAND ----------

model.push_to_hub("tiantan32/bloom-7b1-lora-tagger",
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
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "tiantan32/bloom-7b1-lora-tagger"
config = PeftConfig.from_pretrained(peft_model_id)
model_download = AutoModelForCausalLM.from_pretrained(
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

batch = tokenizer("“Training models with PEFT and LoRa is cool” ->: ", return_tensors='pt')

with torch.cuda.amp.autocast():
  output_tokens = model_download.generate(**batch, max_new_tokens=50)

print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))

# COMMAND ----------

### TO DO
### Load with MLflow
### Train with SamSum conversation cummarization https://huggingface.co/datasets/samsum
### Train with single node multi-GPUs


# COMMAND ----------


