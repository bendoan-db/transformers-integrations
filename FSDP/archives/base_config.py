from torch.distributed.fsdp import (
  FullyShardedDataParallel as FSDP,
  CPUOffload,
  MixedPrecision,
  BackwardPrefetch,
  ShardingStrategy, 
  FullStateDictConfig,
  StateDictType
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
  checkpoint_wrapper,
  CheckpointImpl,
  apply_activation_checkpointing
)
import functools

from dataclasses import dataclass

@dataclass
class base_config:
  verbose: bool = True

  #base training params
  total_steps_to_run: int=5
  batch_size_training: int=15

  #sharding strategy
  sharding_strategy= ShardingStrategy = ShardingStrategy.FULL_SHARD
  print_sharding_plan: bool = False

  run_profiler: bool=False

  backward_prefetch = BackwardPrefetch.BACKWARD_PRE

  #logging
  log_every: int=1
  track_memory=True
  memory_report: bool=True
  nccl_debug_handler:bool=True
  distributed_debug:bool=True

  #dataloaders
  num_workers_dataloder:int=2

  #precision policies
  use_mixed_precision:bool=False
  use_tf32:bool=False

  #activation checkpointing
  fsdp_activation_checkpointing:bool=True

def get_policy_base(blocks):

  recursive_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls=blocks
  )
  return recursive_policy

def fsdp_checkpointing_base(model, blocks):
  """
  apply activation checkpointing to model
  returns None as model is updated directly
  """
  non_reentrant_wrapper = functools.partial(
    checkpoint_wrapper,
    offload_to_cpu=False,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT
  )

  check_fn = lambda submodule: isinstance(submodule, blocks)
  apply_activation_checkpointing_wrapper(
    model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
  )