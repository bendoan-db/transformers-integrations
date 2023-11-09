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
import mlflow
import time
import argparse
import logging
from logging import log
import py7zr
import evaluate
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed, AutoModelForSeq2SeqLM
import datetime

from accelerate import Accelerator, DistributedType
from decimal import *
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
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
model_name = "t5-3b"

def log_gpu_metrics(run_type:str, step):
   for i in range(torch.cuda.device_count()):
     mlflow.log_metric(run_type+"_gpu_utilization_gb_rank_"+str(i)+"_pct", Decimal(torch.cuda.utilization(device=i)/100), step=step)

def get_dataloaders(accelerator: Accelerator, batch_size: int = BATCH_SIZE):
    """
    Creates a set of `DataLoader`s for the `glue` dataset,
    using "bert-base-cased" as the tokenizer.

    Args:
        accelerator (`Accelerator`):
            An `Accelerator` object
        batch_size (`int`, *optional*):
            The batch size for the train and validation DataLoaders.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset('samsum')

    sample_count=0

    def tokenize_function(sample, padding="max_length"):
        # add prefix to the input for t5
      inputs = ["summarize: " + item for item in sample["dialogue"]]

      # tokenize inputs
      # TODO: Dynamically calculate max length of input tokens in dataset
      outputs = tokenizer(inputs, max_length=512, padding=padding, truncation=True)

      # Tokenize targets with the `text_target` keyword argument
      # TODO: Dynamically calculate max length of label tokens in dataset
      labels = tokenizer(text_target=sample["summary"], max_length=95, padding=padding, truncation=True)

      # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
      # padding in the loss.
      if padding == "max_length":
          labels["input_ids"] = [
              [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
          ]

      outputs["labels"] = labels["input_ids"]
      return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    # starting with the main process first:
    with accelerator.main_process_first():
        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["dialogue", "summary", "id"])


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


def training_function(config, args):
    mlflow.end_run()
    with mlflow.start_run():
      mlflow.transformers.autolog()
      # Initialize accelerator
      accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision)
      # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
      lr = config["lr"]
      num_epochs = int(config["num_epochs"])
      seed = int(config["seed"])
      batch_size = int(config["batch_size"])

      metric = evaluate.load("glue", "mrpc")

      # If the batch size is too big we use gradient accumulation
      gradient_accumulation_steps = 1
      if batch_size > MAX_GPU_BATCH_SIZE and accelerator.distributed_type != DistributedType.TPU:
          gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
          batch_size = MAX_GPU_BATCH_SIZE

      set_seed(seed)
      train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size)
      # Instantiate the model (we build the model here so that the seed also control new weights initialization)
      model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
      # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
      # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
      # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
      #model = model.to(accelerator.device)
      # Instantiate optimizer
      optimizer = AdamW(params=model.parameters(), lr=lr)

      # Instantiate scheduler
      lr_scheduler = get_linear_schedule_with_warmup(
          optimizer=optimizer,
          num_warmup_steps=2,
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
          print("\n starting model training..... \n")
          epoch_start_time = time.monotonic()
          model.train()
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
              mlflow.log_metric("loss", loss, step=step)

          #log training metrics        
          log_gpu_metrics("training", epoch)

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
          accelerator.print(f"epoch {epoch}:", eval_metric)

          # log eval metrics
          mlflow.log_metric("epoch_time_minutes", (time.monotonic()-epoch_start_time)/60, step=epoch)
          mlflow.log_metric("accuracy", eval_metric["accuracy"], step=epoch)
          mlflow.log_metric("f1", eval_metric["f1"], step=epoch)
          log_gpu_metrics("eval", epoch)

          mlflow.transformer.log_model(model, artifact_path="dbfs:/Users/ben.doan@databricks.com/models/" +model_name+"_"+ str(int(datetime.datetime.now().timestamp())))


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    args = parser.parse_args()
    config = {"lr": 2e-5, "num_epochs": 5, "seed": 42, "batch_size": BATCH_SIZE}
    training_function(config, args)


if __name__ == "__main__":
    main()