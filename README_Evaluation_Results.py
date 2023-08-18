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
# MAGIC | **Single-GPU**      | TBD       | Training: ~4 hours // Batch Inference: ~2.45 hours   |
# MAGIC | **Multi-GPU**   | 10+ hours then OOM       | TBD      |
# MAGIC
# MAGIC
# MAGIC
# MAGIC ##### Approximate Cost Per Run (Compute + Databricks Licensing)
# MAGIC
# MAGIC |       | **Full Weight** | **PEFT**     |
# MAGIC | :---        |    :----:   |          ---: |
# MAGIC | **Single-GPU**      | XX       | XX   |
# MAGIC | **Multi-GPU**   | XX        | XX      |
# MAGIC
# MAGIC
# MAGIC
# MAGIC ##### Prompt Evaluation

# COMMAND ----------



# COMMAND ----------


