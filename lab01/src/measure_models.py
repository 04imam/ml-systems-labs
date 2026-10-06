import os
import time
import random
import statistics

import numpy as np
import pandas as pd
import psutil
import joblib
import torch

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning

import warnings

# Hide Logistic Regression convergence warning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# -----------------------------
# Reproducibility
# -----------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# -----------------------------
# Load dataset
# -----------------------------
X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# One sample for inference timing
single_sample = X_test[0].reshape(1, -1)

process = psutil.Process(os.getpid())


# -----------------------------
# Model factory
# -----------------------------
def create_logistic():
    return LogisticRegression(
        max_iter=1000,
        random_state=42
    )


def create_random_forest():
    return RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )


# -----------------------------
# Measurement function
# -----------------------------
def measure_model(model_name, model_factory, filename):

    print(f"\nMeasuring {model_name}...")

    # --------------------------------
    # 1. Warm-up training
    # --------------------------------
    warmup_model = model_factory()
    warmup_model.fit(X_train, y_train)

    # --------------------------------
    # 2. Training time - 5 runs
    # --------------------------------
    training_times = []

    for i in range(5):
        model = model_factory()

        start = time.perf_counter()
        model.fit(X_train, y_train)
        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000
        training_times.append(elapsed_ms)

        print(
            f"Training run {i + 1}: "
            f"{elapsed_ms:.4f} ms"
        )

    median_training_time = statistics.median(training_times)

    # --------------------------------
    # Train final model
    # --------------------------------
    final_model = model_factory()

    memory_before_training = process.memory_info().rss

    final_model.fit(X_train, y_train)

    memory_after_training = process.memory_info().rss

    training_memory_bytes = max(
        memory_before_training,
        memory_after_training
    )

    training_memory_mb = (
        training_memory_bytes / (1024 * 1024)
    )

    # --------------------------------
    # 3. Inference - 100 runs
    # --------------------------------
    # Warm-up inference
    final_model.predict(single_sample)

    inference_times = []

    memory_before_inference = process.memory_info().rss

    for _ in range(100):

        start = time.perf_counter()

        final_model.predict(single_sample)

        end = time.perf_counter()

        inference_times.append(
            (end - start) * 1000
        )

    memory_after_inference = process.memory_info().rss

    inference_memory_bytes = max(
        memory_before_inference,
        memory_after_inference
    )

    inference_memory_mb = (
        inference_memory_bytes / (1024 * 1024)
    )

    median_inference_time = statistics.median(
        inference_times
    )

    # --------------------------------
    # 4. Save model
    # --------------------------------
    model_path = os.path.join(
        "lab01",
        "results",
        filename
    )

    joblib.dump(
        final_model,
        model_path
    )

    # --------------------------------
    # 5. Model size
    # --------------------------------
    model_size_bytes = os.path.getsize(model_path)

    model_size_kb = (
        model_size_bytes / 1024
    )

    # --------------------------------
    # Print results
    # --------------------------------
    print(f"\n{model_name} Results")
    print(
        f"Median training time: "
        f"{median_training_time:.4f} ms"
    )
    print(
        f"Median inference time: "
        f"{median_inference_time:.6f} ms"
    )
    print(
        f"Model size: "
        f"{model_size_bytes} bytes"
    )
    print(
        f"Model size: "
        f"{model_size_kb:.2f} KB"
    )
    print(
        f"Training memory: "
        f"{training_memory_mb:.2f} MB"
    )
    print(
        f"Inference memory: "
        f"{inference_memory_mb:.2f} MB"
    )

    return {
        "Model": model_name,
        "Training_Time_Median_ms": round(
            median_training_time, 4
        ),
        "Inference_Time_Median_ms": round(
            median_inference_time, 6
        ),
        "Model_Size_Bytes": model_size_bytes,
        "Model_Size_KB": round(
            model_size_kb, 2
        ),
        "Training_Memory_MB": round(
            training_memory_mb, 2
        ),
        "Inference_Memory_MB": round(
            inference_memory_mb, 2
        )
    }


# -----------------------------
# Measure both models
# -----------------------------
results = []

results.append(
    measure_model(
        "Logistic Regression",
        create_logistic,
        "logistic_model.joblib"
    )
)

results.append(
    measure_model(
        "Random Forest",
        create_random_forest,
        "random_forest_model.joblib"
    )
)

# -----------------------------
# Save measurements
# -----------------------------
results_df = pd.DataFrame(results)

results_df.to_csv(
    "lab01/results/system_measurements.csv",
    index=False
)

print("\n--------------------------------")
print("ALL MEASUREMENTS COMPLETED")
print("--------------------------------")

print(results_df)

print(
    "\nResults saved to:"
    "\nlab01/results/system_measurements.csv"
)