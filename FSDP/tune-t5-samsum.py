# Databricks notebook source
!/databricks/python/bin/pip install py7zr

# COMMAND ----------

# MAGIC %sh accelerate launch --config_file /root/.cache/huggingface/accelerate/default_config.yaml t5-samsum.py

# COMMAND ----------


