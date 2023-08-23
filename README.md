### Experiment: Cost Performance of FSDP, PEFT, and Multi-GPU finetuning

In this experiment, we evaluate the cost performance of training various T5-3B using single GPU and multi - finetuning, with both PEFT and full weight finetuning

- Model: t5-3B
- Dataset: Samsum chat dataset

##### Training Time and Inference

|       | **Full Weight** | **PEFT**     |
| :---        |    :----:   |          ---: |
| **Single-GPU** (A100)     | TBD       | Training: ~4 hours // Batch Inference (800 records): ~2.65 hours   |
| **Multi-GPU** (A10s)  | 10+ hours then OOM       | TBD      |
| **Multi-GPU** (A100s)  | TBD       | TBD      |




##### Approximate Cost Per Run (Compute + Databricks Licensing)

|       | **Full Weight** | **PEFT**     |
| :---        |    :----:   |          ---: |
| **Single-GPU**      | TBD       | TBD    |
| **Multi-GPU**   | TBD        | TBD      |



##### Prompt Evaluation

TO-DO


### Learnings so far (todo:remove for solution accelerator)
- A10s are not suitable for LLM training, even with lots of memory saving optimizations
- The LLM training space is a **mess** in terms of documentation. 
    - There is a severe lack of intuitive documentation that explains how these models work and what parameters you can poke/play with to optimize your models
- `accelerate` provides a fantastic framework for quickly standing up distributed training, specifically Fully Sharded Data Parrellel `FSDP` and `DeepSpeed`
- `PEFT` has potential in terms of memory savings, but at signficant performance cost. More testing is required to evaluate the optimal usage
- If you have an LLM opportunity, **engage an expert**, it's a very dense and there's not a ton of centralized documentation or standardization

### Next Steps
- Test `FSDP` on an A100 cluster
- Test on larger, more effective models (`Llamav2`, `Platypus12B`, `MPT`)
- Integrate with MLflow for better model tracking