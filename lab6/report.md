# Machine Learning Lab 6 — Training Pipelines and Project Proposal

**Student:** Imam Hossain  
**Student ID:** 73947  
**Course:** Machine Learning (CII5SE039CI)  
**Dataset:** FashionMNIST  
**Model:** CNN (20,490 parameters)  
**Device:** CPU

## Task 1 — YAML-Driven Training Pipeline

Reused the CNN developed in Lab 3 and implemented a YAML-configured training pipeline with reproducible random seeds, MLflow experiment tracking, and checkpoint saving.

| Epoch | Training Loss | Validation Accuracy |
|---|---|---|
| 1 | 0.5294 | 84.84% |
| 2 | 0.3509 | 87.04% |
| 3 | 0.3050 | 88.80% |
| 4 | 0.2722 | 89.60% |
| 5 | 0.2609 | 89.74% |

## Task 2 — PyTorch Profiling

| Operation | CPU Time Share |
|---|---|
| Data loading | 43.71% |
| Forward pass | 30.58% |
| Backward pass | 24.17% |
| Optimizer step | 1.53% |

**Conclusion:** Data loading consumed the largest share of measured training time.

## Task 3 — DataLoader Worker Study

| Workers | Throughput (samples/s) |
|---|---|
| 0 | 10377.70 |
| 1 | 4864.13 |
| 2 | 3552.35 |
| 4 | 1376.21 |
| 8 | 625.78 |

**Conclusion:** Zero workers provided the highest throughput in this Windows CPU environment.

## Task 4 — Batch-Size Study

All configurations were trained for five epochs.

| Batch | Throughput (samples/s) | Peak Process RAM (MB) | Validation Accuracy |
|---|---|---|---|
| 32 | 3126.70 | 419.87 | 90.14% |
| 64 | 3528.85 | 428.28 | 90.50% |
| 128 | 3331.02 | 440.30 | 89.70% |
| 256 | 3406.72 | 476.36 | 89.08% |
| 512 | 3268.36 | 521.99 | 88.18% |

**Conclusion:** Batch size 64 delivered the highest throughput and validation accuracy in this study.

## Task 5 — Optimizer and Scheduler Comparison

Each configuration was trained for ten epochs.

| Optimizer | Scheduler | Final Accuracy | Best Accuracy |
|---|---|---|---|
| Adam | StepLR | 91.16% | 91.16% |
| Adam | CosineAnnealingLR | 91.20% | 91.24% |
| Adam | ReduceLROnPlateau | 91.08% | 91.08% |
| SGD | StepLR | 90.62% | 90.96% |
| SGD | CosineAnnealingLR | 90.96% | 90.98% |
| SGD | ReduceLROnPlateau | 90.96% | 91.08% |

**Conclusion:** Adam with CosineAnnealingLR achieved the highest final validation accuracy.

## Task 6 — Gradient Accumulation

| Configuration | Accuracy | Training Time |
|---|---|---|
| Batch 256 | 89.08% | 359.48 s |
| Batch 64, accumulation 4 | 89.28% | 308.61 s |

**Conclusion:** Gradient accumulation approximated the behaviour of a larger batch, with a final accuracy difference of 0.20 percentage points.

## Task 7 — Checkpoint Recovery

Training was resumed from the checkpoint at epoch 3.

| Epoch | Original Accuracy | Resumed Accuracy |
|---|---|---|
| 4 | 89.60% | 89.06% |
| 5 | 89.74% | 89.34% |

**Conclusion:** Model, optimizer, and scheduler states were restored successfully. Exact reproduction remains limited because the DataLoader random generator state was not restored.

## Task 8 — Individual Project Proposal

**Title:** Lightweight CNN for Fashion Image Classification on Edge Devices

The proposal uses FashionMNIST and investigates post-training quantization.

**Planned targets:**
- Test accuracy of at least 90%.
- Inference latency of at most 50 ms.
- Model size of at most 5 MB.

These are proposed targets, not yet verified deployment results.

The proposal is stored in `PROJECT_PROPOSAL.md` and requires instructor approval.

## Evidence Files

- `configs/config.yaml` — YAML configuration
- `train.py` — training and checkpoint recovery
- `profiler_lab6.py` — PyTorch profiling
- `workers_study.py` — DataLoader benchmark
- `batch_study.py` — batch-size study
- `optimizer_study.py` — optimizer and scheduler comparison
- `gradient_accumulation.py` — gradient accumulation
- `results/` — experiment tables, profiler output, and graphs
- `checkpoints/` — saved training checkpoints

## Overall Conclusion

Lab 6 demonstrated a reproducible, configurable CNN training workflow with MLflow tracking, performance profiling, systematic training experiments, gradient accumulation, and checkpoint recovery.

The best final validation accuracy observed in the optimizer and scheduler study was 91.20%.

The individual deployment project is proposed but still requires instructor approval.
