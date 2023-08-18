import time
import tqdm
import torch

from dataclasses import dataclass
from torch.distributed.fsdp import StateDictType

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.models.t5.modeling_t5 import T5Block

#import datasets_grammar as dg
from base_config import base_config, fsdp_checkpointing_base, get_policy_base

@dataclass
class train_config(base_config):
  
  #model
  model_name = "t5-3b"
  tokenizer = "t5-5b"

  #checkpoint models
  save_model_checkpoint:bool=False
  load_model_checkpoint:bool=False
  checkpoint_type = StateDictType.FULL_STATE_DICT
  model_save_name="t5-"
  checkpoint_folder="training_checkpoints"
  checkpoint_max_save_count: int=(
    2
  )

  #optimizers load and save
  save_optimizer:bool=False
  load_optimizer:bool=False
  optimizer_name:str="Adam"

  optimizer_checkpoint_file: str="Adam-t5--1.pt"
  checkpoint_model_filename: str="t5--1.pt"

  #datasets
  dataset_train=

