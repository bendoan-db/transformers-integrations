# Databricks notebook source
dbutils.widgets.text("model_name", "")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

from transformers import AutoTokenizer, T5Model, AutoModelForCausalLM

#TODO: Add logic to pull in pretrained model types

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5Model.from_pretrained(model_name)
print("loaded " + model_name + " as 'model'")
print("loaded " + model_name + "tokenizer as 'tokenizer'")

# COMMAND ----------

# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# orca_tokenizer = AutoTokenizer.from_pretrained("Open-Orca/OpenOrca-Platypus2-13B")
# orca_model = AutoModelForCausalLM.from_pretrained("Open-Orca/OpenOrca-Platypus2-13B")
# print("loaded orca_model, orca_tokenizer")
