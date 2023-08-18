# Databricks notebook source
dbutils.widgets.text("model_name", "")

# COMMAND ----------

username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

model_name = dbutils.widgets.get("model_name")
base_model_path="/Repos/"+username+"/fsdp-transformers-integration/FSDP/" + model_name

model_checkpoint_path=base_model_path+"/model_checkpoints"
model_state_dict_path=base_model_path+"/state_dictionary"

print(base_model_path)
print(model_checkpoint_path)
print(model_state_dict_path)

# COMMAND ----------

import torch
print(torch.__version__)

# COMMAND ----------

# MAGIC %md
# MAGIC ##FSDP: Key Components
# MAGIC **Auto Wrapping**: With FSDP in PyTorch there are two ways to wrap a model - auto-wrapping and manual wrapping. The former is intended to be a drop in replacement for distributed data parallel (a training approach where only data and not the model is sharded for parallelized training across multiple nodes that may have multiple GPUs) whereas the latter provides the flexibility to explore complex sharding strategies. 
# MAGIC
# MAGIC **CPU Offload**: FSDP works by reducing the peak GPU memory required to train a model, allowing training of models that would otherwise not fit into memory. If a model is still too large to fit into memory, wrapped parameters can be offloaded to the CPU when they are not used in computation. This can further improve memory efficiency at the cost of data transfer overhead between host and device.
# MAGIC
# MAGIC **Backward Prefetch**: Backward prefetch is a throughput optimization technique during the backpass phase that controls the timing of a communication step (prefetching the next set of parameters) with a local computation step (gradient computation). Overlapping the communcation and computation steps mean that parameters can begin to be requsted and can arrive sooner at the cost of peak memory usage. It's important to note that parameters need to be fetched since FSDP shards the model 'vertically', meaning that weights in a single layer are sharded across multiple GPUs.
# MAGIC
# MAGIC **FullStateDictConfig**: Similar to streaming applications, training large deep learning models can take weeks or months, leading to an increased (and potentially _very_ expensive) risk of a run stopping unexpectedly making the use of checkpoints almost mandatory. FSDP offers two ways of saving checkpoints. `FullStateDictConfig` aggregates all the parameters from the model from all the GPUs to the CPU, with the assumption that the entire model can fit within CPU memory. The alternative approach is called `LocalStateDict`, saves the state as a collection of sharded files.
# MAGIC
# MAGIC **StateDictType**: `StateDictType` is an enum (a data type that defines a set of predefined constants) that is used to confirm the type of state that is being handled for checkpointing. The enum is defined [here](https://github.com/pytorch/pytorch/blob/8507b22fea9ef819452655aa6dd1c42752a4e74a/torch/distributed/fsdp/api.py#L244-L271).
# MAGIC
# MAGIC **Transformer and Activation Checkpoint Wrapper**: A wrapper class is any class which "wraps" or "encapsulates" the functionality of another class or component. They provide a level of abstraction from the implementation of the underlying class or component. FSDP is implemented as transformer and activation checkpoint wrappers that makes the usage of FSDP simple existing without the need for changing existing model definition code. Activation checkpointing (or gradient checkpointing) is a technique to reduce memory usage by clearing activations of certain layers and recomputing them during a backward pass. 
# MAGIC
# MAGIC !! NEED TO VALIDATE !! Activation checkpointing can be expected to free up 33-38% of GPU memory (allowing the use of larger batch sizes) at the cost of 20-25% training time.
# MAGIC
# MAGIC #### References
# MAGIC 1. https://engineering.fb.com/2021/07/15/open-source/fsdp/
# MAGIC 1. https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/
# MAGIC 1. https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
# MAGIC 1. https://pytorch.org/docs/stable/fsdp.html
# MAGIC 1. https://www.youtube.com/playlist?list=PL_lsbAsL_o2BT6aerEKgIoufVD_fodnuT

# COMMAND ----------

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import (
  FullyShardedDataParallel as FSDP,
  CPUOffload, #? I kinda know what this is but worth rehashing
  MixedPrecision,
  BackwardPrefetch, #?
  ShardingStrategy, 
  FullStateDictConfig, #?
  StateDictType #?
)

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
  #what is a checkpoint wrapper and how is it different from a regular model checkpoint?
  checkpoint_wrapper,
  CheckpointImpl,
  apply_activation_checkpointing #?
)
import functools

# COMMAND ----------

# MAGIC %run ./../load_models $model_name=$model_name

# COMMAND ----------

# DBTITLE 1,Auto Wrapping Policy
from transformers.models.t5.modeling_t5 import T5Block

transformer_auto_wrap_policy = functools.partial(
  transformer_auto_wrap_policy,
  transformer_layer_cls = {
    T5Block,
  }
)

# COMMAND ----------

# DBTITLE 1,Checkpoint Wrapping
#create submodule check function as a lambda
check_fn = lambda submodule: isinstance(submodule, T5Block)

#create non-reentrant wraper to provide options for the checkpoint wrapper for performance
non_reentrant_wrapper = functools.partial(
  checkpoint_wrapper,
  offload_to_cpu = False,
  checkpoint_impl = CheckpointImpl.NO_REENTRANT
)

# COMMAND ----------

# DBTITLE 1,Mixed Precision Policy
from pkg_resources import packaging
import torch.cuda.nccl as nccl
import torch.distributed as dist

bfloat16Policy = MixedPrecision(
  param_dtype=torch.bfloat16,
  reduce_dtype=torch.bfloat16,
  buffer_dtype=torch.bfloat16
)

verify_bfloat16_support = (
  torch.version.cuda
  and torch.cuda.is_bf16_supported()
  and packaging.version.parse(torch.version.cuda).release >=(11,0)
  and dist.is_nccl_available()
  and nccl.version() >= (2,10)
)

assert verify_bfloat16_support

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Model Sharing Strategies
# MAGIC FSDP model sharding strategies balance memory consumption and communication overhead. The there are currently 3 sharding strategies available in FSDP -
# MAGIC 1. No Sharding - Equivalent to the distributed data parallel (DDP) approach where only data sharded and each GPU keeps a full copy of the model
# MAGIC 1. Full Sharding - This is the default strategy wherein the model parameters, gradients and optimizer state are all sharded meaning that memory utilization is minimized. Testing has shown that on the same hardware, a fully sharded model can have 3.33x the number of parameters a DDP model can.
# MAGIC 1. `SHARD_GRAD_OP` - Which is a contraction of 'shard gradients and optimizer' is in between full sharding and no sharding. Specifically the model parameters are not purged from memory after the forward pass but rather after the backwards pass.
# MAGIC
# MAGIC A future hybrid sharding approach will perform full sharding on each node but will not sharding between each node since inter-node communications are slower than intra-node communication.

# COMMAND ----------

# DBTITLE 1,Model Sharding Strategy
#@krish - further documentation here would be very good

#balancing memory vs. communication overhead, accoutning for sharding of params, optimizer states, and gradients
ShardingStrategy.FULL_SHARD #default - optimize all model states, optimizers, and gradients .. maximize model size support
ShardingStrategy.SHARD_GRAD_OP #model params 
ShardingStrategy.NO_SHARD #DDP mode - mimimize communication overhead, but minimal model size support

# COMMAND ----------

# MAGIC %md
# MAGIC ## Backwards Prefetch
# MAGIC 1. None: Don't pass anything summon next FSDP unit after the `all_reduce` for current layer gradients. All params are dropped, gradients are calculated and scattered first
# MAGIC 2. `BackwardPrefetch.BACKWARD_POST`: Prefetch next params after the params are dropped but before the gradients are scattered
# MAGIC 3. `BackwardPrefetch.BACKWARD_PRE`: Prefetch params at the start of the FSDP unit computation. Basically, current and next params are fetched at the same time. Adds to peak memory
# MAGIC     - Testing indicates 13% speed up with 0.59% memory increase

# COMMAND ----------

# MAGIC %md
# MAGIC ## Child Finetuning
# MAGIC - Task Free vs. Task Dependent
# MAGIC - Applies mask to subset of parameters to be updated. Updated parameter gradients will be magnified to improve generalizability

# COMMAND ----------

#TO: Fix later and figure out how to actually bring this file in the right way
spark.sparkContext.addPyFile("./Workspace/Repos/ben.doan@databricks.com/fsdp-transformers-integration/FSDP/ChildTuningOptimizer.py")

import ChildTuningOptimizer
from ChildTuningOptimizer import ChildTuningAdamW
reserve_p = 0.3

# COMMAND ----------

ct_optimizer = ChildTuningAdamW(model.parameters(), lr=4e-8, reserve_p=reserve_p, mode="taskfree")

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

loss = output["loss"]

if scaler: #?
  scaler.scale(loss)
  scaler.scale(optimizer)
  scaler.update() #adjusting scaling for each minibatch

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Config and Checkpointing
#if (t5_3b_model.config and t5_3b_model.config.checkpoint_type == StateDictType.FULL_STATE_DICT):
#  print("yuh")
PATH = "/Repos/ben.doan@databricks.com/fsdp-transformers-integration/FSDP/checkpoint_test"
torch.save(jsonObject, PATH)

# COMMAND ----------

config = torch.load(PATH)
#config.checkpoint_type

# COMMAND ----------

torch.save(t5_3b_model.state_dict(), "/Repos/ben.doan@databricks.com/fsdp-transformers-integration/FSDP/state_dict.pt")

# COMMAND ----------

state_dict = torch.load("/Repos/ben.doan@databricks.com/fsdp-transformers-integration/FSDP/state_dict.pt")
"checkpoint_type" in checkpoint.keys()

# COMMAND ----------

t5_3b_model.load_state_dict(state_dict)

# COMMAND ----------

optimizer = torch.optim.AdamW(
  t5_3b_model.parameters(), lr=8e-4, weight_decay=0.005
)

# COMMAND ----------

fsdp_model = FSDP(
  t5_3b_model,
  auto_wrap_policy=transformer_auto_wrap_policy,
  mixed_precision=bfloat16Policy,
  sharding_strategy=ShardingStrategy.FULL_SHARD,
  device_id=torch.cuda.current_device(),
  forward_prefetch=True,
  backward_prefetch = BackwardPrefetch.BACKWARD_PRE
)

apply_activation_checkpointing_wrapper(
  model, checkpoint_wrapper=non_reentrant_wrapper, check_fn=check_fn
)

# COMMAND ----------

samsum_dataset

# COMMAND ----------

# MAGIC %sh accelerate launch nlp_example.py

# COMMAND ----------


