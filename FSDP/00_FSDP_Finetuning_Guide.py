# Databricks notebook source
# MAGIC %md
# MAGIC # What is Fully Sharded Data Parallel (FSDP)?
# MAGIC
# MAGIC PyTorch's Fully Sharded Data Parallelism (FSDP) is a powerful tool for such distributed training. It's designed to reduce the memory footprint required for each GPU, allowing for training even larger models than was previously feasible. Here's an overview of the key concepts:
# MAGIC
# MAGIC ### Model Sharding
# MAGIC
# MAGIC - The core idea behind FSDP is to shard (split) the model's parameters and optimizer states across all GPUs, so that each GPU only stores a fraction of the total. This drastically reduces the per-GPU memory requirements.
# MAGIC - In the forward pass, each GPU only needs a shard of the model, and computes only a part of the total computation.
# MAGIC - During the backward pass, gradients are computed for the shard specific to each GPU.
# MAGIC - In this process, gradients need to be synchronized across GPUs after the backward pass so that all GPUs have a consistent view. This is done using collective communication operations (e.g., all-reduce).
# MAGIC  - FSDP employs techniques like gradient accumulation to reduce the communication overhead, by synchronizing gradients less frequently than every iteration.
# MAGIC
# MAGIC ### CPU Offload
# MAGIC
# MAGIC - To squeeze even more memory savings, FSDP offers the option to offload optimizer states and gradients to the CPU memory.
# MAGIC - This is beneficial when GPU memory is a major bottleneck, but it does come with the trade-off of increased computation time due to data transfers between CPU and GPU.
# MAGIC
# MAGIC ### Ease of Use
# MAGIC - FSDP is designed to be as plug-and-play as possible. The goal is to wrap an existing PyTorch model with FSDP with minimal changes to the existing codebase.
# MAGIC - This means you can convert your single-GPU code to use FSDP with minimal hassle, and benefit from distributed training capabilities.
# MAGIC
# MAGIC #### Benefits
# MAGIC
# MAGIC - Allows training larger models that don’t fit in GPU memory.
# MAGIC - Potentially faster training times through efficient parallelism and reduced communication overhead.
# MAGIC - Simplified code changes when transitioning from single-GPU to multi-GPU training.
# MAGIC #### Drawbacks
# MAGIC - More complex than standard DataParallel or DistributedDataParallel approaches in PyTorch.
# MAGIC - May require fine-tuning and understanding of the underlying principles for best performance.
# MAGIC
# MAGIC In summary, Fully Sharded Data Parallelism in PyTorch is a sophisticated tool for distributed deep learning, allowing for the training of models that were previously too large to handle, both in terms of memory and computation. By sharding the model across multiple GPUs and employing other optimization techniques, FSDP provides a scalable and efficient solution for deep learning at scale

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

# MAGIC %md
# MAGIC
# MAGIC ## FSDP Model Sharing Strategies
# MAGIC FSDP model sharding strategies balance memory consumption and communication overhead. The there are currently 3 sharding strategies available in FSDP -
# MAGIC 1. No Sharding - Equivalent to the distributed data parallel (DDP) approach where only data sharded and each GPU keeps a full copy of the model
# MAGIC 1. Full Sharding - This is the default strategy wherein the model parameters, gradients and optimizer state are all sharded meaning that memory utilization is minimized. Testing has shown that on the same hardware, a fully sharded model can have 3.33x the number of parameters a DDP model can.
# MAGIC 1. `SHARD_GRAD_OP` - Which is a contraction of 'shard gradients and optimizer' is in between full sharding and no sharding. Specifically the model parameters are not purged from memory after the forward pass but rather after the backwards pass.
# MAGIC
# MAGIC A future hybrid sharding approach will perform full sharding on each node but will not sharding between each node since inter-node communications are slower than intra-node communication.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Backwards Prefetch Strategies
# MAGIC 1. None: Don't pass anything summon next FSDP unit after the `all_reduce` for current layer gradients. All params are dropped, gradients are calculated and scattered first
# MAGIC 2. `BackwardPrefetch.BACKWARD_POST`: Prefetch next params after the params are dropped but before the gradients are scattered
# MAGIC 3. `BackwardPrefetch.BACKWARD_PRE`: Prefetch params at the start of the FSDP unit computation. Basically, current and next params are fetched at the same time. Adds to peak memory
# MAGIC     - Testing indicates 13% speed up with 0.59% memory increase

# COMMAND ----------


