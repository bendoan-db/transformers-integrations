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


