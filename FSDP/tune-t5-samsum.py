# Databricks notebook source
!/databricks/python/bin/pip install py7zr

# COMMAND ----------

# MAGIC %sh accelerate launch --config_file /Workspace/Repos/ben.doan@databricks.com/fsdp-transformers-integration/FSDP/config/t5-fsdp-config.yaml t5-samsum.py

# COMMAND ----------


