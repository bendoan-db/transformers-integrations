# Databricks notebook source
dbutils.fs.cp("file:///databricks/driver/llama/", "dbfs:/Users/ben.doan@databricks.com/models/llama2", recurse=True)

# COMMAND ----------

# MAGIC %%bash
# MAGIC # pip install llama-recipes transformers datasets accelerate sentencepiece protobuf==3.20 py7zr scipy peft bitsandbytes fire torch_tb_profiler ipywidgets
# MAGIC %sh TRANSFORM=`python -c "import transformers;print('/'.join(transformers.__file__.split('/')[:-1])+'/models/llama/convert_llama_weights_to_hf.py')"`
# MAGIC # python ${TRANSFORM} --input_dir models --model_size 7B --output_dir models_hf/7B

# COMMAND ----------

!/databricks/python/bin/pip install llama-recipes transformers datasets accelerate sentencepiece protobuf==3.20 py7zr scipy peft bitsandbytes fire torch_tb_profiler flash-attn

# COMMAND ----------

# MAGIC %sh 
# MAGIC pip install llama-recipes transformers datasets accelerate sentencepiece protobuf==3.20 py7zr scipy peft bitsandbytes fire torch_tb_profiler ipywidgets
# MAGIC pip install --upgrade flash-attn>=2.0
# MAGIC TRANSFORM=`python -c "import transformers;print('/Workspace/Repos/ben.doan@databricks.com/transformers-integrations/FSDP/models/llama2/convert_llama_weights_to_hf.py')"`
# MAGIC python ${TRANSFORM} --input_dir /dbfs/Users/ben.doan@databricks.com/models/llama2/llama-2-7b --model_size 7B --output_dir /dbfs/Users/ben.doan@databricks.com/models/llama-2-7b-hf

# COMMAND ----------

TOKEN="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"

# COMMAND ----------

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
model_path = "dbfs:/Users/ben.doan@databricks.com/models/llama2"
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=TOKEN)
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", token=TOKEN)

# COMMAND ----------

from llama_recipes.utils.dataset_utils import get_preprocessed_dataset
from llama_recipes.configs.datasets import samsum_dataset

llama_dataset = get_preprocessed_dataset(tokenizer, samsum_dataset).train_test_split(test_size=0.3)
#split_dataset = llama_dataset.train_test_split(test_size=0.3)

# COMMAND ----------

llama_dataset["train"]

# COMMAND ----------

from datasets import load_dataset

dataset = load_dataset("samsum", split="train")
dataset[:1]

# COMMAND ----------


