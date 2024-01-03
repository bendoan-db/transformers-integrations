# Databricks notebook source
# MAGIC %pip install --upgrade mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import time
import evaluate
import mlflow
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from accelerate import Accelerator, DistributedType
from decimal import *
from datetime import datetime
import pynvml


########################################################################
# This is a fully working simple example to use Accelerate
#
# This example trains a Bert base model on GLUE MRPC
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - (multi) TPUs
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, follow the instructions
# in the readme for examples:
# https://github.com/huggingface/accelerate/tree/main/examples
#
########################################################################


MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
model_name = "bert"


def get_dataloaders(accelerator: Accelerator, batch_size: int = 16):
    """
    Creates a set of `DataLoader`s for the `glue` dataset,
    using "bert-base-cased" as the tokenizer.

    Args:
        accelerator (`Accelerator`):
            An `Accelerator` object
        batch_size (`int`, *optional*):
            The batch size for the train and validation DataLoaders.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    datasets = load_dataset("glue", "mrpc")

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    # starting with the main process first:
    with accelerator.main_process_first():
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", "sentence1", "sentence2"],
        )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    def collate_fn(examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        max_length = 128 if accelerator.distributed_type == DistributedType.TPU else None
        # When using mixed precision we want round multiples of 8/16
        if accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None

        return tokenizer.pad(
            examples,
            padding="longest",
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size, drop_last=True
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=EVAL_BATCH_SIZE,
        drop_last=(accelerator.mixed_precision == "fp8"),
    )

    return train_dataloader, eval_dataloader

def log_gpu_metrics(run_type:str, step):
   for i in range(torch.cuda.device_count()):
     mlflow.log_metric(run_type+"_gpu_utilization_gb_rank_"+str(i)+"_pct", Decimal(torch.cuda.utilization(device=i)/100), step=step)

def set_up_mlflow(experiment_name):
    """
    Set up the MLflow experiment. If the experiment exists, it will use it.
    Otherwise, it creates a new experiment with the provided name.

    Args:
    experiment_name (str): The name of the MLflow experiment

    Returns:
    str: The experiment ID of the existing or newly created experiment
    """
    # Check if the experiment already exists
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is not None:
        # Experiment exists, use the existing experiment ID
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
    else:
        # Experiment does not exist, create a new one
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")

    return experiment_id

def training_function(mixed_precision="fp16", seed: int = 42, batch_size: int = 64, epochs=5, lr=3e-2 / 25, cpu=False, experiment_name=None):

    #TODO: Setup logic for autogenerating experiment names
    experiment_id = set_up_mlflow(experiment_name)
    mlflow.end_run(experiment_id=experiment_id)
    with mlflow.start_run(experiment_id=experiment_id):
      # Initialize accelerator
      accelerator = Accelerator(cpu=cpu, mixed_precision=mixed_precision, log_with="all")
      # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
      lr = lr
      num_epochs = epochs
      seed = seed
      batch_size = batch_size

      metric = evaluate.load("glue", "mrpc")

      # If the batch size is too big we use gradient accumulation
      gradient_accumulation_steps = 1
      if batch_size > MAX_GPU_BATCH_SIZE and accelerator.distributed_type != DistributedType.TPU:
          gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
          batch_size = MAX_GPU_BATCH_SIZE

      set_seed(seed)
      train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size)
      # Instantiate the model (we build the model here so that the seed also control new weights initialization)
      model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", return_dict=True)

      # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
      # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
      # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
      model = model.to(accelerator.device)
      # Instantiate optimizer
      optimizer = AdamW(params=model.parameters(), lr=lr)

      # Instantiate scheduler
      lr_scheduler = get_linear_schedule_with_warmup(
          optimizer=optimizer,
          num_warmup_steps=100,
          num_training_steps=(len(train_dataloader) * num_epochs) // gradient_accumulation_steps,
      )

      # Prepare everything
      # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
      # prepare method.

      model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
          model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
      )

      # Now we train the model
      for epoch in range(num_epochs):
          model.train()
          epoch_start_time = time.monotonic()
          for step, batch in enumerate(train_dataloader):
              # We could avoid this line since we set the accelerator with `device_placement=True`.
              batch.to(accelerator.device)
              outputs = model(**batch)
              loss = outputs.loss
              loss = loss / gradient_accumulation_steps
              accelerator.backward(loss)
              if step % gradient_accumulation_steps == 0:
                  optimizer.step()
                  lr_scheduler.step()
                  optimizer.zero_grad()
          
          log_gpu_metrics("training", epoch)
          mlflow.log_metric("loss", loss, step=epoch)

          model.eval()
          for step, batch in enumerate(eval_dataloader):
              # We could avoid this line since we set the accelerator with `device_placement=True`.
              batch.to(accelerator.device)
              with torch.no_grad():
                  outputs = model(**batch)
              predictions = outputs.logits.argmax(dim=-1)
              predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
              metric.add_batch(
                  predictions=predictions,
                  references=references,
              )

          eval_metric = metric.compute()
          # Use accelerator.print to print only on the main process.
          mlflow.log_metric("epoch_time_minutes", (time.monotonic()-epoch_start_time)/60, step=epoch)
          mlflow.log_metric("accuracy", eval_metric["accuracy"], step=epoch)
          mlflow.log_metric("f1", eval_metric["f1"], step=epoch)
          log_gpu_metrics("eval", epoch)
          accelerator.print(f"epoch {epoch}:", eval_metric)
    
    mlflow_model_components = {"model": accelerate.utils.extract_model_from_parallel(model), "tokenizer": AutoTokenizer.from_pretrained("bert-base-cased")}

    mlflow.transformers.log_model(mlflow_model_components, "doan_bert_model")

# COMMAND ----------

from accelerate import notebook_launcher
import os
args = ("fp16", 42, 64, 5, 3e-2 / 25, False, "/Users/ben.doan@databricks.com/doan_test/doan_bert_llm_tuning")
os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

notebook_launcher(training_function, args, num_processes=2)

# COMMAND ----------



# COMMAND ----------

test_id = set_up_mlflow("/Users/ben.doan@databricks.com/doan_test")
print(test_id)

# COMMAND ----------


