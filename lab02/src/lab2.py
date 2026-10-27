
# ============================================================
# LAB 2: NEURAL NETWORK FROM SCRATCH
# Tasks 1-7 | UCI / Scikit-learn Digits Dataset
# ============================================================

from pathlib import Path
from time import perf_counter
import csv

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT PATHS AND SETTINGS
# ============================================================

LAB_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = LAB_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
HIDDEN_WIDTHS = [16, 64]
LEARNING_RATES = [0.001, 0.01, 0.1]
EPOCHS = 50
BATCH_SIZE = 32


# ============================================================
# TASK 1: DATASET PREPARATION
# ============================================================

print("\nTASK 1: DATASET PREPARATION")

digits = load_digits()

X = digits.data.astype(np.float64) / 16.0
y = digits.target

# One-hot encode labels
Y = np.eye(10)[y]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE
)

print("Training:", X_train.shape)
print("Testing:", X_test.shape)

assert X_train.shape[1] == 64
assert y_train.shape[1] == 10

print("Task 1: PASSED")


# ============================================================
# TASK 2: TWO-HIDDEN-LAYER NETWORK ARCHITECTURE
# ============================================================

print("\nTASK 2: NETWORK ARCHITECTURE")


def initialize_parameters(width, seed=42, method="normal"):
    rng = np.random.default_rng(seed)

    if method == "he":
        s1 = np.sqrt(2.0 / 64)
        s2 = np.sqrt(2.0 / width)
        s3 = np.sqrt(2.0 / width)
    else:
        s1 = s2 = s3 = 0.01

    params = {
        "W1": rng.normal(0, s1, (64, width)),
        "b1": np.zeros(width),
        "W2": rng.normal(0, s2, (width, width)),
        "b2": np.zeros(width),
        "W3": rng.normal(0, s3, (width, 10)),
        "b3": np.zeros(10)
    }

    return params


for width in HIDDEN_WIDTHS:
    params = initialize_parameters(width)

    print(f"\nW={width}")

    for name, value in params.items():
        print(name, value.shape)

    assert params["W1"].shape == (64, width)
    assert params["W2"].shape == (width, width)
    assert params["W3"].shape == (width, 10)

print("\nTask 2: PASSED")


# ============================================================
# TASK 3: FORWARD PROPAGATION
# ============================================================

def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(np.float64)


def softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_values = np.exp(shifted)

    return exp_values / np.sum(
        exp_values,
        axis=1,
        keepdims=True
    )


def forward(X, params):
    Z1 = X @ params["W1"] + params["b1"]
    A1 = relu(Z1)

    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = relu(Z2)

    Z3 = A2 @ params["W3"] + params["b3"]
    A3 = softmax(Z3)

    cache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
        "Z3": Z3,
        "A3": A3
    }

    return A3, cache


print("\nTASK 3: FORWARD PROPAGATION")

for width in HIDDEN_WIDTHS:
    params = initialize_parameters(width)
    predictions, cache = forward(X_train[:32], params)

    assert predictions.shape == (32, 10)
    assert np.allclose(
        predictions.sum(axis=1),
        1.0,
        atol=1e-6
    )

    print(f"W={width}: Output shape {predictions.shape}")

print("Task 3: PASSED")


# ============================================================
# TASK 4: CROSS-ENTROPY LOSS
# ============================================================

def cross_entropy_loss(y_true, predictions):
    clipped = np.clip(predictions, 1e-15, 1.0)

    return -np.mean(
        np.sum(y_true * np.log(clipped), axis=1)
    )


print("\nTASK 4: CROSS-ENTROPY LOSS")

for width in HIDDEN_WIDTHS:
    params = initialize_parameters(width)
    predictions, _ = forward(X_train, params)

    loss = cross_entropy_loss(y_train, predictions)

    print(f"W={width}: Loss = {loss:.6f}")

    assert abs(loss - np.log(10)) < 0.01

print("Expected untrained loss:", np.log(10))
print("Task 4: PASSED")


# ============================================================
# TASK 5: BACKPROPAGATION
# ============================================================

def backward(y_true, params, cache):
    X = cache["X"]
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    Z2 = cache["Z2"]
    A2 = cache["A2"]
    A3 = cache["A3"]

    m = X.shape[0]

    # Output layer
    dZ3 = (A3 - y_true) / m

    dW3 = A2.T @ dZ3
    db3 = np.sum(dZ3, axis=0)

    # Second hidden layer
    dA2 = dZ3 @ params["W3"].T
    dZ2 = dA2 * relu_derivative(Z2)

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)

    # First hidden layer
    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * relu_derivative(Z1)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
        "W3": dW3,
        "b3": db3
    }


print("\nTASK 5: BACKPROPAGATION")

params = initialize_parameters(16, method="he")
predictions, cache = forward(X_train[:32], params)

grads = backward(y_train[:32], params, cache)

for name in params:
    assert grads[name].shape == params[name].shape

print("Gradient dimensions: PASSED")


# ============================================================
# NUMERICAL GRADIENT CHECKING
# ============================================================

print("\nNUMERICAL GRADIENT CHECKING")

EPSILON = 1e-5
THRESHOLD = 1e-6
rng = np.random.default_rng(42)

# Small fixed batch for reproducible gradient checking
X_check = X_train[:8]
y_check = y_train[:8]

params = initialize_parameters(16, method="he")

predictions, cache = forward(X_check, params)
analytical_grads = backward(y_check, params, cache)

passed = 0
errors = []

parameter_names = list(params.keys())

for i in range(10):
    name = parameter_names[i % len(parameter_names)]
    parameter = params[name]

    index = tuple(
        int(rng.integers(size))
        for size in parameter.shape
    )

    original_value = parameter[index]

    # Positive perturbation
    parameter[index] = original_value + EPSILON
    pred_plus, _ = forward(X_check, params)
    loss_plus = cross_entropy_loss(y_check, pred_plus)

    # Negative perturbation
    parameter[index] = original_value - EPSILON
    pred_minus, _ = forward(X_check, params)
    loss_minus = cross_entropy_loss(y_check, pred_minus)

    # Restore original weight
    parameter[index] = original_value

    numerical = (
        loss_plus - loss_minus
    ) / (2 * EPSILON)

    analytical = analytical_grads[name][index]

    relative_error = abs(
        numerical - analytical
    ) / max(
        1e-8,
        abs(numerical) + abs(analytical)
    )

    errors.append(relative_error)

    if relative_error < THRESHOLD:
        passed += 1
        status = "PASS"
    else:
        status = "FAIL"

    print(
        f"{name}{index}: "
        f"Error={relative_error:.3e} [{status}]"
    )

print("-" * 55)
print(f"Gradient checks passed: {passed}/10")
print(f"Maximum relative error: {max(errors):.3e}")

assert passed >= 9, "Gradient checking failed"

print("Task 5: PASSED")


# ============================================================
# TASK 6: MINI-BATCH TRAINING
# ============================================================

def calculate_accuracy(X, y_true, params):
    predictions, _ = forward(X, params)

    predicted_classes = np.argmax(
        predictions, axis=1
    )

    actual_classes = np.argmax(
        y_true, axis=1
    )

    return np.mean(
        predicted_classes == actual_classes
    )


def train_network(
    width,
    epochs=50,
    learning_rate=0.1,
    batch_size=32
):
    # He initialization supports training with ReLU.
    params = initialize_parameters(
        width,
        seed=42,
        method="he"
    )

    rng = np.random.default_rng(42)
    loss_history = []

    for epoch in range(epochs):
        indices = rng.permutation(len(X_train))

        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for start in range(
            0,
            len(X_shuffled),
            batch_size
        ):
            X_batch = X_shuffled[
                start:start + batch_size
            ]

            y_batch = y_shuffled[
                start:start + batch_size
            ]

            predictions, cache = forward(
                X_batch, params
            )

            grads = backward(
                y_batch, params, cache
            )

            for name in params:
                params[name] -= (
                    learning_rate * grads[name]
                )

        predictions, _ = forward(X_train, params)

        loss = cross_entropy_loss(
            y_train,
            predictions
        )

        loss_history.append(float(loss))

        print(
            f"W={width} | LR={learning_rate} | "
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss={loss:.4f}"
        )

    train_accuracy = calculate_accuracy(
        X_train,
        y_train,
        params
    )

    test_accuracy = calculate_accuracy(
        X_test,
        y_test,
        params
    )

    return (
        params,
        loss_history,
        train_accuracy,
        test_accuracy
    )


print("\nTASK 6: TRAINING NEURAL NETWORKS")

results = {}

for width in HIDDEN_WIDTHS:
    for learning_rate in LEARNING_RATES:
        print(
            f"\nTraining W={width}, "
            f"LR={learning_rate}"
        )

        (
            params,
            losses,
            train_acc,
            test_acc
        ) = train_network(
            width=width,
            epochs=EPOCHS,
            learning_rate=learning_rate,
            batch_size=BATCH_SIZE
        )

        results[(width, learning_rate)] = {
            "losses": losses,
            "train_acc": float(train_acc),
            "test_acc": float(test_acc)
        }

        print(
            f"Training accuracy: "
            f"{train_acc * 100:.2f}%"
        )

        print(
            f"Testing accuracy: "
            f"{test_acc * 100:.2f}%"
        )


# ============================================================
# TASK 6: LOSS CURVES
# ============================================================

plt.figure(figsize=(12, 7))

for (width, learning_rate), result in results.items():
    plt.plot(
        range(1, EPOCHS + 1),
        result["losses"],
        label=f"W={width}, LR={learning_rate}"
    )

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Lab 2 - Neural Network Training Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "loss_curves.png",
    dpi=300
)

plt.close()

print("\nLoss curves saved to results/loss_curves.png")


# ============================================================
# TASK 6: ACCURACY COMPARISON
# ============================================================

print("\nTASK 6: ACCURACY COMPARISON")
print("-" * 68)

print(
    f"{'Width':<10}"
    f"{'Learning Rate':<18}"
    f"{'Train Accuracy':<18}"
    f"{'Test Accuracy'}"
)

for (width, learning_rate), result in results.items():
    print(
        f"{width:<10}"
        f"{learning_rate:<18}"
        f"{result['train_acc'] * 100:<18.2f}"
        f"{result['test_acc'] * 100:.2f}"
    )

with open(
    RESULTS_DIR / "task6_accuracy_results.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.writer(file)

    writer.writerow([
        "width",
        "learning_rate",
        "epochs",
        "batch_size",
        "train_accuracy",
        "test_accuracy"
    ])

    for (width, lr), result in results.items():
        writer.writerow([
            width,
            lr,
            EPOCHS,
            BATCH_SIZE,
            result["train_acc"],
            result["test_acc"]
        ])

print("\nTask 6: COMPLETED")


# ============================================================
# TASK 7: FORWARD / BACKWARD TIMING
# ============================================================

print("\nTASK 7: EXECUTION TIME BENCHMARKING")


def benchmark_network(width, repetitions=100):
    params = initialize_parameters(
        width,
        method="he"
    )

    X_batch = X_train[:32]
    y_batch = y_train[:32]

    forward_times = []
    backward_times = []

    # Warm-up
    for _ in range(5):
        predictions, cache = forward(X_batch, params)
        backward(y_batch, params, cache)

    for _ in range(repetitions):
        start = perf_counter()

        predictions, cache = forward(
            X_batch, params
        )

        forward_ms = (
            perf_counter() - start
        ) * 1000

        start = perf_counter()

        grads = backward(
            y_batch,
            params,
            cache
        )

        backward_ms = (
            perf_counter() - start
        ) * 1000

        forward_times.append(forward_ms)
        backward_times.append(backward_ms)

    forward_median = float(
        np.median(forward_times)
    )

    backward_median = float(
        np.median(backward_times)
    )

    return {
        "width": width,
        "forward_ms": forward_median,
        "backward_ms": backward_median,
        "total_ms": forward_median + backward_median
    }


benchmark_results = []

for width in HIDDEN_WIDTHS:
    result = benchmark_network(
        width,
        repetitions=100
    )

    benchmark_results.append(result)

    print(f"\nNetwork width: {width}")

    print(
        f"Forward median: "
        f"{result['forward_ms']:.6f} ms"
    )

    print(
        f"Backward median: "
        f"{result['backward_ms']:.6f} ms"
    )

    print(
        f"Total: "
        f"{result['total_ms']:.6f} ms"
    )


# ============================================================
# TASK 7: SAVE BENCHMARK CSV
# ============================================================

with open(
    RESULTS_DIR / "task7_benchmark_results.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.writer(file)

    writer.writerow([
        "width",
        "forward_ms",
        "backward_ms",
        "total_ms"
    ])

    for result in benchmark_results:
        writer.writerow([
            result["width"],
            result["forward_ms"],
            result["backward_ms"],
            result["total_ms"]
        ])

print("\nBenchmark CSV saved.")


# ============================================================
# TASK 7: EXECUTION TIME CHART
# ============================================================

widths = [
    result["width"]
    for result in benchmark_results
]

forward_values = [
    result["forward_ms"]
    for result in benchmark_results
]

backward_values = [
    result["backward_ms"]
    for result in benchmark_results
]

positions = np.arange(len(widths))
bar_width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    positions - bar_width / 2,
    forward_values,
    bar_width,
    label="Forward"
)

plt.bar(
    positions + bar_width / 2,
    backward_values,
    bar_width,
    label="Backward"
)

plt.xticks(positions, [str(w) for w in widths])
plt.xlabel("Hidden Layer Width")
plt.ylabel("Median Execution Time (ms)")
plt.title("Forward vs Backward Pass Timing")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "task7_execution_time.png",
    dpi=300
)

plt.close()

for result in benchmark_results:
    dominant = (
        "Backward"
        if result["backward_ms"] > result["forward_ms"]
        else "Forward"
    )

    print(
        f"W={result['width']}: "
        f"{dominant} pass takes longer."
    )

print(
    "\nMatrix multiplication dominates computation. "
    "Backpropagation typically performs more matrix "
    "multiplications because it must calculate both "
    "parameter gradients and upstream gradients."
)

print("\nTask 7: COMPLETED")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("LAB 2 COMPLETED SUCCESSFULLY")
print("========================================")

print("Task 1: PASSED")
print("Task 2: PASSED")
print("Task 3: PASSED")
print("Task 4: PASSED")
print(f"Task 5: {passed}/10 GRADIENT CHECKS PASSED")
print("Task 6: COMPLETED")
print("Task 7: COMPLETED")

best_configuration = max(
    results,
    key=lambda key: results[key]["test_acc"]
)

best_result = results[best_configuration]

print("\nBEST CONFIGURATION")
print("Width:", best_configuration[0])
print("Learning rate:", best_configuration[1])
print(
    "Test accuracy:",
    f"{best_result['test_acc'] * 100:.2f}%"
)

print("\nGenerated result files:")

for path in sorted(RESULTS_DIR.iterdir()):
    print("-", path.name)

print("\nLAB 2 EXECUTION FINISHED")
