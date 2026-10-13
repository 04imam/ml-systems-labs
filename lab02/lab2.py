
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ==========================================
# TASK 1: DATASET PREPARATION
# ==========================================

digits = load_digits()
X = digits.data / 16.0
y = digits.target
Y = np.eye(10)[y]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTASK 1: DATASET PREPARATION")
print("Training:", X_train.shape)
print("Testing:", X_test.shape)

assert X_train.shape == (1257, 64)
assert X_test.shape == (540, 64)
assert y_train.shape == (1257, 10)

print("Task 1: PASSED")


# ==========================================
# TASK 2: NETWORK INITIALIZATION
# ==========================================

def initialize_parameters(width, seed=42, method="small"):
    rng = np.random.default_rng(seed)

    if method == "he":
        s1 = np.sqrt(2.0 / 64)
        s2 = np.sqrt(2.0 / width)
        s3 = np.sqrt(1.0 / width)
    elif method == "small":
        s1 = s2 = s3 = 0.01
    else:
        raise ValueError("Unknown initialization method")

    return {
        "W1": rng.normal(0, s1, (64, width)),
        "b1": np.zeros(width),
        "W2": rng.normal(0, s2, (width, width)),
        "b2": np.zeros(width),
        "W3": rng.normal(0, s3, (width, 10)),
        "b3": np.zeros(10)
    }


print("\nTASK 2: NETWORK ARCHITECTURE")

for width in [16, 64]:
    params = initialize_parameters(width)

    print(f"W={width}")
    for name, value in params.items():
        print(name, value.shape)

    assert params["W1"].shape == (64, width)
    assert params["W2"].shape == (width, width)
    assert params["W3"].shape == (width, 10)

print("Task 2: PASSED")


# ==========================================
# TASK 3: FORWARD PROPAGATION
# ==========================================

def relu(x):
    return np.maximum(0, x)


def softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


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
        "A3": A3
    }

    return A3, cache


print("\nTASK 3: FORWARD PROPAGATION")

for width in [16, 64]:
    params = initialize_parameters(width)
    predictions, cache = forward(X_train, params)

    assert predictions.shape == (1257, 10)
    assert np.allclose(predictions.sum(axis=1), 1.0)

    print(f"W={width}: Output {predictions.shape}")

print("Task 3: PASSED")


# ==========================================
# TASK 4: CROSS-ENTROPY LOSS
# ==========================================

def cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1.0)

    return -np.mean(
        np.sum(y_true * np.log(y_pred), axis=1)
    )


print("\nTASK 4: CROSS-ENTROPY LOSS")

expected_loss = np.log(10)

for width in [16, 64]:
    params = initialize_parameters(width)
    predictions, _ = forward(X_train, params)

    loss = cross_entropy_loss(y_train, predictions)

    print(f"W={width}: Loss = {loss:.6f}")
    assert abs(loss - expected_loss) < 0.1

print("Expected untrained loss:", expected_loss)
print("Task 4: PASSED")


# ==========================================
# TASK 5: BACKPROPAGATION
# ==========================================

def backward(y_true, params, cache):
    X = cache["X"]
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    Z2 = cache["Z2"]
    A2 = cache["A2"]
    A3 = cache["A3"]

    m = X.shape[0]

    # Output layer gradient
    dZ3 = (A3 - y_true) / m
    dW3 = A2.T @ dZ3
    db3 = np.sum(dZ3, axis=0)

    # Second hidden layer gradient
    dA2 = dZ3 @ params["W3"].T
    dZ2 = dA2 * (Z2 > 0)

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)

    # First hidden layer gradient
    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * (Z1 > 0)

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


# ==========================================
# NUMERICAL GRADIENT CHECKING
# ==========================================

def gradient_check(X_batch, y_batch, params,
                   epsilon=1e-5):

    predictions, cache = forward(X_batch, params)
    analytical_grads = backward(y_batch, params, cache)

    rng = np.random.default_rng(123)

    # Check all parameter groups
    parameter_names = [
        "W1", "b1", "W2", "b2", "W3",
        "b3", "W1", "W2", "W3", "b2"
    ]

    errors = []

    print("\nNUMERICAL GRADIENT CHECKING")
    print("-" * 55)

    for name in parameter_names:
        tensor = params[name]
        analytical_tensor = analytical_grads[name]

        # Select a coordinate with a measurable gradient.
        # This avoids division instability near zero.
        candidates = np.argwhere(
            np.abs(analytical_tensor) > 1e-4
        )

        if len(candidates) == 0:
            raise ValueError(
                f"No suitable gradient for {name}"
            )

        index = tuple(
            candidates[rng.integers(len(candidates))]
        )

        original = tensor[index]

        # Positive perturbation
        tensor[index] = original + epsilon
        pred_plus, _ = forward(X_batch, params)
        loss_plus = cross_entropy_loss(
            y_batch, pred_plus
        )

        # Negative perturbation
        tensor[index] = original - epsilon
        pred_minus, _ = forward(X_batch, params)
        loss_minus = cross_entropy_loss(
            y_batch, pred_minus
        )

        # Restore parameter
        tensor[index] = original

        numerical = (
            loss_plus - loss_minus
        ) / (2 * epsilon)

        analytical = analytical_tensor[index]

        relative_error = abs(
            numerical - analytical
        ) / max(
            1e-8,
            abs(numerical) + abs(analytical)
        )

        errors.append(relative_error)

        status = (
            "PASS" if relative_error < 1e-6
            else "FAIL"
        )

        print(
            f"{name}{index}: "
            f"Error={relative_error:.3e} "
            f"[{status}]"
        )

    passed = sum(error < 1e-6 for error in errors)

    print("-" * 55)
    print(f"Gradient checks passed: {passed}/10")
    print(f"Maximum relative error: {max(errors):.3e}")

    return passed


print("\nTASK 5: BACKPROPAGATION")

X_batch = X_train[:20]
y_batch = y_train[:20]

# Use He initialization for meaningful gradient checks.
params = initialize_parameters(
    16,
    seed=42,
    method="he"
)

predictions, cache = forward(X_batch, params)
grads = backward(y_batch, params, cache)

for name in params:
    assert grads[name].shape == params[name].shape
    assert np.all(np.isfinite(grads[name]))

print("Gradient dimensions: PASSED")

checks_passed = gradient_check(
    X_batch,
    y_batch,
    params,
    epsilon=1e-5
)

assert checks_passed >= 9, (
    "Gradient check failed: fewer than 9/10 passed"
)

print("Task 5: PASSED")


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n===================================")
print("LAB 2: NEURAL NETWORK FROM SCRATCH")
print("===================================")

for task in range(1, 6):
    print(f"Task {task}: PASSED")

print("===================================")

# ==========================================
# TASK 6: TRAINING AND COMPARISON
# ==========================================

import matplotlib.pyplot as plt

def calculate_accuracy(X, y_true, params):
    predictions, _ = forward(X, params)

    predicted_classes = np.argmax(predictions, axis=1)
    actual_classes = np.argmax(y_true, axis=1)

    return np.mean(predicted_classes == actual_classes)


def train_network(width, epochs=200, learning_rate=0.1):
    params = initialize_parameters(
        width, seed=42, method="he"
    )

    loss_history = []

    for epoch in range(epochs):
        # Forward propagation
        predictions, cache = forward(X_train, params)

        # Calculate loss
        loss = cross_entropy_loss(y_train, predictions)
        loss_history.append(loss)

        # Backpropagation
        grads = backward(y_train, params, cache)

        # Gradient descent update
        for name in params:
            params[name] -= learning_rate * grads[name]

        if (epoch + 1) % 20 == 0:
            print(
                f"W={width} | Epoch {epoch+1}/{epochs} "
                f"| Loss={loss:.4f}"
            )

    train_acc = calculate_accuracy(
        X_train, y_train, params
    )
    test_acc = calculate_accuracy(
        X_test, y_test, params
    )

    return params, loss_history, train_acc, test_acc


print("\nTASK 6: TRAINING NEURAL NETWORKS")

results = {}

for width in [16, 64]:
    print(f"\nTraining network with W={width}")

    params, losses, train_acc, test_acc = train_network(
        width,
        epochs=200,
        learning_rate=0.1
    )

    results[width] = {
        "losses": losses,
        "train_acc": train_acc,
        "test_acc": test_acc
    }

    print(f"Training accuracy: {train_acc * 100:.2f}%")
    print(f"Testing accuracy: {test_acc * 100:.2f}%")


# Plot training losses
plt.figure(figsize=(10, 6))

for width in [16, 64]:
    plt.plot(
        results[width]["losses"],
        label=f"Hidden width = {width}"
    )

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "task6_training_loss.png",
    dpi=300
)

plt.show()


print("\nTASK 6: ACCURACY COMPARISON")
print("-" * 48)
print(f"{'Width':<10}{'Train Accuracy':<20}{'Test Accuracy'}")

for width in [16, 64]:
    result = results[width]

    print(
        f"{width:<10}"
        f"{result['train_acc']*100:<20.2f}"
        f"{result['test_acc']*100:.2f}"
    )

print("\nTask 6: COMPLETED")

# ==========================================
# TASK 7: EXECUTION TIME BENCHMARKING
# ==========================================

import time
import csv
import matplotlib.pyplot as plt


def benchmark_network(width, repetitions=100):
    # Initialize the network
    params = initialize_parameters(
        width,
        seed=42,
        method="he"
    )

    # Use the same training dataset for both widths
    X_batch = X_train
    y_batch = y_train

    forward_times = []
    backward_times = []

    # Warm-up runs
    for _ in range(5):
        predictions, cache = forward(X_batch, params)
        backward(y_batch, params, cache)

    # Benchmark forward and backward passes
    for _ in range(repetitions):

        # Measure forward propagation
        start = time.perf_counter()

        predictions, cache = forward(X_batch, params)

        end = time.perf_counter()
        forward_times.append((end - start) * 1000)

        # Measure backpropagation
        start = time.perf_counter()

        grads = backward(y_batch, params, cache)

        end = time.perf_counter()
        backward_times.append((end - start) * 1000)

    avg_forward = np.mean(forward_times)
    avg_backward = np.mean(backward_times)

    return {
        "width": width,
        "forward_ms": avg_forward,
        "backward_ms": avg_backward,
        "total_ms": avg_forward + avg_backward
    }


print("\n===================================")
print("TASK 7: EXECUTION TIME BENCHMARKING")
print("===================================")

benchmark_results = []

for width in [16, 64]:
    result = benchmark_network(
        width,
        repetitions=100
    )

    benchmark_results.append(result)

    print(f"\nNetwork width: {width}")
    print(
        f"Average forward time: "
        f"{result['forward_ms']:.4f} ms"
    )
    print(
        f"Average backward time: "
        f"{result['backward_ms']:.4f} ms"
    )
    print(
        f"Total computation time: "
        f"{result['total_ms']:.4f} ms"
    )


# ==========================================
# SAVE RESULTS TO CSV
# ==========================================

with open(
    "task7_benchmark_results.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Network Width",
        "Forward Time (ms)",
        "Backward Time (ms)",
        "Total Time (ms)"
    ])

    for result in benchmark_results:
        writer.writerow([
            result["width"],
            result["forward_ms"],
            result["backward_ms"],
            result["total_ms"]
        ])

print("\nBenchmark results saved to CSV.")


# ==========================================
# PERFORMANCE COMPARISON GRAPH
# ==========================================

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

x = np.arange(len(widths))
bar_width = 0.35

plt.figure(figsize=(9, 5))

plt.bar(
    x - bar_width / 2,
    forward_values,
    bar_width,
    label="Forward Pass"
)

plt.bar(
    x + bar_width / 2,
    backward_values,
    bar_width,
    label="Backward Pass"
)

plt.xlabel("Hidden Layer Width")
plt.ylabel("Average Execution Time (ms)")
plt.title("Forward vs Backward Execution Time")

plt.xticks(
    x,
    [f"W={width}" for width in widths]
)

plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "task7_execution_time.png",
    dpi=300
)

plt.show()


# ==========================================
# FINAL LAB SUMMARY
# ==========================================

print("\n===================================")
print("LAB 2: FINAL SUMMARY")
print("===================================")

for task in range(1, 8):
    print(f"Task {task}: COMPLETED")

print("\nGenerated files:")
print("task6_training_loss.png")
print("task7_execution_time.png")
print("task7_benchmark_results.csv")

print("\nLAB 2 EXECUTION FINISHED")
