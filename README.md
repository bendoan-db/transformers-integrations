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


### Next Steps
- Deploy onto an A100 cluster
- Test on Llamav2`, `Platypus12B`, `MPT`
- Integrate with MLflow
- Benchmark performance of FSDP vs. FSDP PEFT
