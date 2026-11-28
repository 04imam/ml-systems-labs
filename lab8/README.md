# Lab 8 - Quantization and Export

Student: Imam Hossain
Student ID: 73947

## Completed experiments

1. Manual INT8, INT4 and INT2 quantization.
2. Post-training dynamic quantization.
3. Static INT8 quantization using 500 calibration images.
4. ONNX export and numerical validation.
5. Combined pruning, distillation and quantization.
6. GTSRB individual project baseline.

## Run experiments

From the repository root:

    python lab8/src/manual_quantization.py
    python lab8/src/dynamic_quantization.py
    python lab8/src/static_quantization.py
    python lab8/src/onnx_export.py
    python lab8/src/combined_optimization.py
    python lab8/project/train_baseline.py

## Requirements

Lab 7 teacher, structured pruning and distilled student checkpoints
must be available locally. Regenerate them using Lab 7 if necessary.

The quantization backend is oneDNN because FBGEMM is unavailable
in the installed PyTorch build.

## Individual project

Real-Time Traffic Sign Recognition for Driver-Assistance
on a CPU-Only Edge Device.

Dataset: GTSRB
Architecture: Pretrained ResNet-18
Classes: 43
Test accuracy: 95.94%
Macro-F1: 0.9475
Median CPU latency: 14.83 ms

The trained baseline checkpoint is included in project/checkpoints.

## Results

See report.md and the results directories for experiment metrics,
graphs, numerical validation and deployment recommendations.
