
import time
import statistics
import numpy as np
import torch
from sklearn.datasets import make_moons
from autograd_engine import Value

torch.set_num_threads(1)

# Same dataset and initial weights for both implementations
X, y = make_moons(n_samples=60, noise=0.15, random_state=42)
X = X.astype(np.float64)
y = y.astype(np.float64).reshape(-1, 1)

EPOCHS = 60
LR = 0.1
RUNS = 5

rng = np.random.default_rng(42)
W1 = rng.normal(0, 0.5, (2, 8))
b1 = np.zeros(8)
W2 = rng.normal(0, 0.5, (8, 1))
b2 = np.zeros(1)


def train_custom():
    w1 = [[Value(v) for v in row] for row in W1]
    bias1 = [Value(v) for v in b1]
    w2 = [Value(v) for v in W2[:, 0]]
    bias2 = Value(b2[0])

    params = [p for row in w1 for p in row]
    params += bias1 + w2 + [bias2]

    start = time.perf_counter()

    for epoch in range(EPOCHS):
        losses = []

        for x, target in zip(X, y[:, 0]):
            hidden = []

            for j in range(8):
                z = bias1[j]
                for i in range(2):
                    z = z + w1[i][j] * float(x[i])
                hidden.append(z.tanh())

            z = bias2
            for j in range(8):
                z = z + hidden[j] * w2[j]

            prediction = z.sigmoid()

            # Binary cross-entropy
            loss = -(
                float(target) * prediction.log()
                + (1.0 - float(target)) * (1 - prediction).log()
            )
            losses.append(loss)

        total_loss = sum(losses) / len(losses)
        total_loss.backward()

        for p in params:
            p.data -= LR * p.grad

    elapsed = time.perf_counter() - start
    return elapsed, total_loss.data


def train_pytorch():
    x = torch.tensor(X, dtype=torch.float64)
    targets = torch.tensor(y, dtype=torch.float64)

    w1 = torch.nn.Parameter(torch.tensor(W1.copy()))
    bias1 = torch.nn.Parameter(torch.tensor(b1.copy()))
    w2 = torch.nn.Parameter(torch.tensor(W2.copy()))
    bias2 = torch.nn.Parameter(torch.tensor(b2.copy()))

    params = [w1, bias1, w2, bias2]

    start = time.perf_counter()

    for epoch in range(EPOCHS):
        for p in params:
            p.grad = None

        hidden = torch.tanh(x @ w1 + bias1)
        predictions = torch.sigmoid(hidden @ w2 + bias2)

        loss = torch.nn.functional.binary_cross_entropy(
            predictions, targets
        )
        loss.backward()

        with torch.no_grad():
            for p in params:
                p -= LR * p.grad

    elapsed = time.perf_counter() - start
    return elapsed, loss.item()


custom_times = []
torch_times = []

for run in range(RUNS):
    custom_time, custom_loss = train_custom()
    torch_time, torch_loss = train_pytorch()

    custom_times.append(custom_time)
    torch_times.append(torch_time)

    print(f"\nRun {run + 1}")
    print(f"Custom: {custom_time:.4f}s, loss={custom_loss:.6f}")
    print(f"PyTorch: {torch_time:.4f}s, loss={torch_loss:.6f}")

custom_median = statistics.median(custom_times)
torch_median = statistics.median(torch_times)

print("\n--- FINAL COMPARISON ---")
print(f"Custom median: {custom_median:.4f}s")
print(f"PyTorch median: {torch_median:.4f}s")
print(f"Custom/PyTorch time ratio: {custom_median / torch_median:.2f}x")
print("TASK 5 BENCHMARK COMPLETED")
