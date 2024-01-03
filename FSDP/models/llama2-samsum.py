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
import argparse
import logging
from logging import log
import py7zr
import evaluate
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed, AutoModelForSeq2SeqLM, LlamaForCausalLM, LlamaTokenizer
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset
from llama_recipes.configs.datasets import samsum_dataset

from accelerate import Accelerator, DistributedType
from decimal import *
from datetime import datetime
import time
import pynvml
from model_utilities.model_utils import get_hf_token, log_gpu_metrics_mlflow

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


MAX_GPU_BATCH_SIZE = 2
BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
TOKEN=get_hf_token(use_env_variable=True)
model_name="meta-llama/Llama-2-7b-hf"


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

    tokenizer = LlamaTokenizer.from_pretrained(model_name, token=TOKEN)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    dataset = get_preprocessed_dataset(tokenizer, samsum_dataset).train_test_split(test_size=0.3)
    

    def collate_fn(examples):
      # When using mixed precision we want round multiples of 8/16
      max_length = 1024
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
        dataset["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size, drop_last=True
    )

    print(str(len(dataset["train"][0]["input_ids"])))

    eval_dataloader = DataLoader(
        dataset["test"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=EVAL_BATCH_SIZE,
        drop_last=(accelerator.mixed_precision == "fp8"),
    )

    return train_dataloader, eval_dataloader

def training_function(config, args):
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
    
    # Instantiate the model
    device = accelerator.device
    model = LlamaForCausalLM.from_pretrained(model_name, token=TOKEN, torch_dtype=torch.float16)
    model.to(device)
    model = accelerator.prepare(model)
    
    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=3,
        num_training_steps=(len(train_dataloader) * num_epochs) // gradient_accumulation_steps,
    )

    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.

    optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    with mlflow.start_run():
      # Now we train the model
      for epoch in range(num_epochs):
          model.train()
          epoch_start_time = time.monotonic()
          print(torch.cuda.memory_summary(device=None, abbreviated=False))
          for step, batch in enumerate(train_dataloader):
              print("step: " + str(step))
              print(len(batch["input_ids"][0]))
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
          # Use accelerator.print to print only on the main process.
          accelerator.print(f"epoch {epoch}:", eval_metric)
    
    mlflow_model_components = {"model": accelerate.utils.extract_model_from_parallel(model), "tokenizer": LlamaTokenizer.from_pretrained(model_name, token=TOKEN)}

    mlflow.transformers.log_model(mlflow_model_components, "doan_llama27b_model")

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