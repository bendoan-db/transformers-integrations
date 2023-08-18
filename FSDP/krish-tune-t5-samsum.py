# Databricks notebook source
!/databricks/python3/bin/pip install py7zr

# COMMAND ----------

# MAGIC %md
# MAGIC ####Experiment 1: Turn off CPU offloading####
# MAGIC Result OOM

# COMMAND ----------

# MAGIC %sh accelerate launch --config_file /root/.cache/huggingface/accelerate/default_config.yaml t5-samsum.py

# COMMAND ----------

# MAGIC %md
# MAGIC ####Experiment 2: Keep CPU offloading off and reduce batch size to 2 from 4####

# COMMAND ----------

# MAGIC %sh accelerate launch --config_file /root/.cache/huggingface/accelerate/default_config.yaml krish-t5-samsum.py

# COMMAND ----------

# MAGIC %md ####Experiment 3: Keep CPU offloading off and reduce batch size to 1 from 2####

# COMMAND ----------

# MAGIC %sh accelerate launch --config_file /root/.cache/huggingface/accelerate/default_config.yaml krish-t5-samsum.py

# COMMAND ----------

# MAGIC %md ####[WRONG] Experiment 4: Turn CPU offloading on and increase batch size to 8####
# MAGIC Success! Still only max 20% GPU memory utilization

# COMMAND ----------

# MAGIC %sh accelerate launch --config_file /root/.cache/huggingface/accelerate/default_config.yaml t5-samsum.py

# COMMAND ----------

# MAGIC %md ####Experiment 5: Turn CPU offloading off and increase batch size to 3####

# COMMAND ----------

# MAGIC %sh accelerate launch --config_file /root/.cache/huggingface/accelerate/default_config.yaml krish-t5-samsum.py

# COMMAND ----------

# MAGIC %md ####Experiment 6: Turn CPU offloading on and increase batch size to 32####

# COMMAND ----------

# MAGIC %sh accelerate launch --config_file /root/.cache/huggingface/accelerate/default_config.yaml krish-t5-samsum.py

# COMMAND ----------

# MAGIC %md ####Experiment 7: Turn CPU offloading on and increase batch size to 16####

# COMMAND ----------

# MAGIC %sh accelerate launch --config_file /root/.cache/huggingface/accelerate/default_config.yaml krish-t5-samsum.py

# COMMAND ----------

# MAGIC %md ####Experiment 8: Turn CPU offloading on and increase batch size to 8####

# COMMAND ----------

# MAGIC %sh accelerate launch --config_file /root/.cache/huggingface/accelerate/default_config.yaml krish-t5-samsum.py

# COMMAND ----------

# MAGIC %md ####Experiment 9: Turn CPU offloading on and increase batch size to 6####

# COMMAND ----------

# MAGIC %sh accelerate launch --config_file /root/.cache/huggingface/accelerate/default_config.yaml krish-t5-samsum.py
