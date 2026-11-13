# Individual Machine Learning Project Proposal

Student: Imam Hossain
Student ID: 73947
Course: Machine Learning (CII5SE039CI)

## Project Title
Lightweight CNN for Fashion Image Classification on Edge Devices

## Problem
Image classification models can require significant computational resources.
This project aims to develop a small and efficient CNN that classifies
clothing images accurately while remaining suitable for edge devices.

## Dataset
FashionMNIST, containing 70,000 grayscale images of clothing
across 10 categories, each with a resolution of 28 x 28 pixels.

Training: 55,000 images
Validation: 5,000 images
Testing: 10,000 images

## Proposed Method
Use the CNN architecture developed in Lab 3 as the baseline.
Train using PyTorch with reproducible YAML configurations.
Track experiments using MLflow and compare training settings.

## Success Metrics
Target classification accuracy: at least 90% on the test set.
Measure inference latency, model size, and classification accuracy.

## Deployment Constraints
Inference latency: <= 50 milliseconds per image.
Model size: <= 5 MB.
Target environment: CPU-based edge device.

## Planned Optimization
Apply post-training quantization and compare the optimized model
against the original CNN.

Measure changes in accuracy, latency, and model size.

## Expected Outcome
A reproducible and optimized image classification model that
meets the deployment constraints with minimal accuracy loss.

## Tools
Python, PyTorch, torchvision, MLflow, PyYAML, NumPy,
pandas, and Matplotlib.
