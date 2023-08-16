# Databricks notebook source
from transformers import AutoTokenizer, T5Model
t5_tokenizer = AutoTokenizer.from_pretrained("t5-3b")
t5_3b_model = T5Model.from_pretrained("t5-3b")
print("loaded t5_3b_model, t5_tokenizer")

# COMMAND ----------

# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# orca_tokenizer = AutoTokenizer.from_pretrained("Open-Orca/OpenOrca-Platypus2-13B")
# orca_model = AutoModelForCausalLM.from_pretrained("Open-Orca/OpenOrca-Platypus2-13B")
# print("loaded orca_model, orca_tokenizer")
