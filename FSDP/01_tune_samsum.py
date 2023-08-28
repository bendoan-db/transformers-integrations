# Databricks notebook source
# MAGIC %md
# MAGIC # Tuning Models on the Samsum Dataset
# MAGIC
# MAGIC To tune models on the samsum dataset, you will need a model training file (in the `models` directory), as well as an `accelerate` configuration yaml file in the `config` directory. 
# MAGIC
# MAGIC To launch the accelerate job, you will need to point the `%sh` command to your model file and config file, as shown below

# COMMAND ----------

import os
import json

os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ['MLFLOW_EXPERIMENT_NAME'] = "/Users/ben.doan@databricks.com/fine-tuning-t5-samsum"
os.environ['MLFLOW_FLATTEN_PARAMS'] = "true"

host = json.loads(dbutils.notebook.entry_point.getDbutils().notebook() \
  .getContext().toJson())['tags']['browserHostName']

os.environ['DATABRICKS_HOST'] = "https://" + host

# COMMAND ----------

# DBTITLE 1,Install any required libraries
!/databricks/python/bin/pip install py7zr

# COMMAND ----------

# DBTITLE 1,Launch accelerate job
# MAGIC %sh export DATABRICKS_TOKEN && export DATABRICKS_HOST && export MLFLOW_EXPERIMENT_NAME && export MLFLOW_FLATTEN_PARAMS=true && accelerate launch --config_file /Workspace/Repos/ben.doan@databricks.com/transformers-integrations/FSDP/config/t5-fsdp-config-cpu.yaml ./models/t5-samsum.py

# COMMAND ----------


