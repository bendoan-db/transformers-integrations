# Databricks notebook source
import torch
print(torch.__version__)

# COMMAND ----------

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import (
  FullyShardedDataParallel as FSDP,
  CPUOffload,
  MixedPrecision,
  BackwardPrefetch,
  ShardingStrategy,
  FullStateDictConfig,
  StateDictType
)

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
  checkpoint_wrapper,
  CheckpointImpl,
  apply_activation_checkpointing
)
import functools

# COMMAND ----------

# MAGIC %run ./../load_models

# COMMAND ----------

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
non_reentrant_wrapper = partial(
  checkpoint_wrapper,
  offload_to_cpu = False,
  checkpoint_impl = CheckpointImpl.NO_REENTRANT
)

# COMMAND ----------

bfloat16Policy = MixedPrecision(
  param_dtype=torch.bfloat16,
  reduce_dtype=torch.bfloat16,
  buffer_dtype=torch.bfloat16
)

# COMMAND ----------

model = FSDP(
  orca_model,
  auto_wrap_policy=transformer_auto_wrap_policy,
  mixed_precision=bfloat16Policy,
  #sharding_strategy=model_sharding_strategy,
  device_id=torch.cuda.current_device()
)

apply_activation_checkpointing_wrapper(
  model, checkpoint_wrapper=non_reentrant_wrapper, check_fn=check_fn
)

# COMMAND ----------


