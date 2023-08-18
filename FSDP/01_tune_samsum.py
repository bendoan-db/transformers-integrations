# Databricks notebook source
# MAGIC %md
# MAGIC # Tuning Models on the Samsum Dataset
# MAGIC
# MAGIC To tune models on the samsum dataset, you will need a model training file (in the `models` directory), as well as an `accelerate` configuration yaml file in the `config` directory. 
# MAGIC
# MAGIC To launch the accelerate job, you will need to point the `%sh` command to your model file and config file, as shown below

# COMMAND ----------

# DBTITLE 1,Install any required libraries
!/databricks/python/bin/pip install py7zr

# COMMAND ----------

# DBTITLE 1,Launch accelerate job
# MAGIC %sh accelerate launch --config_file /Workspace/Repos/ben.doan@databricks.com/fsdp-transformers-integration/FSDP/config/t5-fsdp-config.yaml ./models/t5-samsum.py

# COMMAND ----------


