
import time
import statistics
import numpy as np
import torch
from sklearn.datasets import make_moons

# Ensure repeatable experiments
np.random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(1)

X, y = make_moons(n_samples=200, noise=0.15, random_state=42)

X = torch.tensor(X, dtype=torch.float64)
y = torch.tensor(y.reshape(-1, 1), dtype=torch.float64)

def train_once():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 1),
        torch.nn.Sigmoid()
    ).double()

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    start = time.perf_counter()

    for epoch in range(300):
        optimizer.zero_grad()
        predictions = model(X)
        loss = loss_fn(predictions, y)
        loss.backward()
        optimizer.step()

    elapsed = time.perf_counter() - start

    with torch.no_grad():
        predictions = model(X)
        accuracy = ((predictions >= 0.5) == y).double().mean().item()

    return elapsed, accuracy, loss.item()

times = []
accuracies = []

for run in range(5):
    elapsed, accuracy, loss = train_once()
    times.append(elapsed)
    accuracies.append(accuracy)

    print(
        f"Run {run + 1}: "
        f"Time={elapsed:.4f}s, "
        f"Accuracy={accuracy:.2%}, "
        f"Loss={loss:.6f}"
    )

print("\nMedian training time:", round(statistics.median(times), 4), "seconds")
print("Mean accuracy:", round(statistics.mean(accuracies) * 100, 2), "%")
