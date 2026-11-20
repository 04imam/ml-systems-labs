# Individual Project Proposal

**Student:** Imam Hossain  **ID:** 73947
**Course:** Machine Learning (CII5SE039CI), Lab 6, Task 8

## Title
Real-Time Traffic Sign Recognition for Driver-Assistance on a CPU-Only Edge Device

## Problem
Modern cars read road signs (speed limits, stop, no entry) with a small camera computer that has no GPU, little memory and a strict time budget. A large vision model recognises signs well but is too big and slow to run on every video frame. This project builds a strong traffic-sign classifier and then compresses it so it fits an in-car edge device, while staying reliable in bad conditions such as night, glare, blur and fog.

This is harder than my earlier labs (digits, Fashion-MNIST, Adult). The images are real colour photos taken outdoors, there are 43 classes with strong class imbalance, and a wrong prediction (for example a 30 km/h sign read as 80 km/h) has real safety consequences, so error analysis matters, not just accuracy.

## Dataset
German Traffic Sign Recognition Benchmark (GTSRB): about 39,000 training and 12,600 test colour images across 43 sign classes, with large variation in size, lighting and viewpoint (`torchvision.datasets.GTSRB`). I will hold out a validation split from the training set, grouped by physical sign track so near-identical frames of the same sign do not leak into validation. Images are resized to 32x32 or 48x48. The imbalanced classes will be handled with weighted sampling or loss weights, and I will report per-class results.

## Success metrics
- Test accuracy of at least 97% and macro-F1 of at least 0.95 on the official test set.
- Accuracy drop of at most 1.5 percentage points versus the uncompressed teacher.
- Robustness: a measured accuracy drop under synthetic night/darkness, motion blur, fog and noise corruptions, reported per corruption, with a target drop of at most 8 points.
- Safety analysis: a confusion study of speed-limit classes, with the most dangerous confusions reported separately.

## Deployment constraints
- Model size of at most 5 MB (target under 1 MB after quantization).
- Latency of at most 50 ms per image at batch size 1 on CPU, including resize and normalisation (median and p95 over 100 runs). I will also report the latency headroom, since a real camera needs several frames per second.

## Models
- **Teacher:** a pretrained ResNet-18 fine-tuned on GTSRB (about 45 MB).
- **Student:** a small custom CNN or a width-reduced MobileNet-style network (under 1 MB in FP32).

## Planned optimization (all three, compared and combined)
1. **Knowledge distillation:** train the student on soft teacher outputs, with a temperature and loss-weight sweep, versus training on hard labels only.
2. **Structured pruning:** remove channels at 30%, 50% and 70% with fine-tuning.
3. **Quantization:** post-training static INT8 versus quantization-aware training.

Ablation table: FP32 baseline, then +distillation, +pruning, +quantization, and the full combination. For each row I will record accuracy, macro-F1, size in MB, latency and corruption robustness. The final result is an accuracy-versus-latency-versus-size trade-off plot, using the Lab 3 cost approach (hand-calculated FLOPs versus profiled).

## Engineering (reusing Lab 6 infrastructure)
Every run is driven by one YAML config with fixed seeds, checkpointing and MLflow logging of params, metrics and artifacts. `torch.profiler` shows whether data loading, preprocessing or the network is the bottleneck, and the `num_workers` and batch-size studies are reused. Everything is committed with a README and a one-command run script.

## Risks and fallback
If INT8 quantization hurts accuracy too much, I will use quantization-aware training or keep sensitive layers in FP32. If time is short, the minimum scope is distillation, one pruning level and INT8 quantization with the ablation table.
