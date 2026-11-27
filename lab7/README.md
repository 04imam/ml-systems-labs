# ML Lab 7 — Compression: Pruning and Distillation

**Student:** Imam Hossain | **ID:** 73947  
**Course:** Machine Learning (CII5SE039CI)

Covers **all five graded tasks** from the supplied Lab 7 manual, as one experiment suite.

## Run (Windows PowerShell)

```powershell
cd "$HOME\OneDrive\Desktop\ml-systems-labs\lab7"
.\run_lab7.ps1
```

Or install `requirements.txt` and run `python src/run_lab7.py`. The runner automatically downloads Fashion-MNIST on first use (internet required). For a quick structural check, `python src/run_lab7.py --smoke` runs only a reduced grid (NOT a full submission).

## Task coverage

1. CNN with approximately one million trainable parameters on Fashion-MNIST; evaluate test accuracy, file size, MACs and CPU latency.
2. Global and layerwise exact-magnitude masking at 0, 25, 50, 75, 90, 95% sparsity, mask-frozen three-epoch fine-tuning; structured L2 filter removal at 10, 25, 50%, followed by three epochs of fine-tuning; sparsity curves.
3. Time all models using warmed batch-1 CPU inference, including resize and normalization; compare original checkpoints, gzip-compressed dense checkpoints and sparse-weights `.npz` representations. The latter is a storage-only, non-deployable representation.
4. Train a ~100k parameter student with normal cross entropy and with teacher-guided KL distillation, T=4 and alpha=0.5; compare accuracy, size and latency.
5. Automatically generate `report.md` with a metrics table and the four efficiency pillars (data, model, compute, energy), plus an evidence-based conclusion.

## Output

- `report.md`: actual test results, documented protocol and conclusions (only after experiment finishes).
- `results/metrics.csv`, `results/metrics.json`: experimental measurements.
- `results/sparsity_curves.png`: pruning accuracy curves.
- `results/distillation_comparison.png`: plain vs distilled student.
- `results/accuracy_latency_size.png`: accuracy / CPU latency / size relationship.
- `results/training_histories.json`: epoch-level train loss and validation accuracy.
- `checkpoints/`: each model checkpoint (can be many MB).

## Notes about measurement

- Fashion-MNIST train/validation split is 54,000/6,000. Test set is 10,000.
- Benchmark uses CPU irrespective of training device. Hardware-dependent results should be run on the submission computer.
- MACs are counted through forward hooks (conv / linear only), approximate FLOPs = 2×MACs.
- Electrical energy **is not measured** without a power meter. MACs and timing are only proxies.
- Formula for distillation uses standard `KL(teacher || student)` via PyTorch `kl_div(log_student, probs_teacher)`, which differs in argument order from the manual's shorthand `KL(student || teacher)`; this is intentional and disclosed.
- Do **not** submit `--smoke` results as a finished full experiment.
