# Databricks notebook source
# MAGIC %md
# MAGIC # Tuning Models on the Samsum Dataset
# MAGIC
# MAGIC To tune models on the samsum dataset, you will need a model training file (in the `models` directory), as well as an `accelerate` configuration yaml file in the `config` directory. 
# MAGIC
# MAGIC To launch the accelerate job, you will need to point the `%sh` command to your model file and config file, as shown below

# COMMAND ----------

# DBTITLE 1,Install any required libraries
!/databricks/python/bin/pip install torch==2.1.0
!/databricks/python/bin/pip install -U accelerate
!/databricks/python/bin/pip install mlflow==2.7.1
!/databricks/python/bin/pip install pynvml 
!/databricks/python/bin/pip install --upgrade flash-attn>=2.0

# COMMAND ----------

!/databricks/python/bin/pip install llama-recipes datasets accelerate sentencepiece protobuf==3.20 scipy peft bitsandbytes fire torch_tb_profiler flash-attn

# COMMAND ----------

from pynvml import *
import os

os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ['MLFLOW_EXPERIMENT_NAME'] = "/Users/ben.doan@databricks.com/doan-accelerate-t5-samsum"
os.environ['MLFLOW_FLATTEN_PARAMS'] = "true"

#check to make sure gpu mem isn't being used by any other processes
def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

print_gpu_utilization()

# COMMAND ----------

# DBTITLE 1,Launch accelerate job
# MAGIC %sh export DATABRICKS_TOKEN && export DATABRICKS_HOST && export MLFLOW_EXPERIMENT_NAME && export MLFLOW_FLATTEN_PARAMS=true && accelerate launch --config_file /Workspace/Repos/ben.doan@databricks.com/transformers-integrations/FSDP/config/llama2-fsdp-config.yaml ./models/llama2-samsum.py

# COMMAND ----------


