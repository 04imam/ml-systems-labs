
import time
import statistics
import os
import torch

torch.manual_seed(42)
torch.set_num_threads(1)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


model = SmallModel().eval()
x = torch.randn(1024, 2)

# Trace the model
traced = torch.jit.trace(model, x)

print("=== MODEL GRAPH ===")
print(traced.inlined_graph)

print("\n=== TENSOR INFORMATION ===")
print("Shape:", x.shape)
print("Dtype:", x.dtype)
print("Contiguous:", x.is_contiguous())


def benchmark(fn, iterations=1000):
    # Warm-up before measurement
    with torch.no_grad():
        for _ in range(100):
            fn()

        timings = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(iterations):
                fn()
            timings.append(
                (time.perf_counter() - start) / iterations * 1000
            )

    return statistics.median(timings)


eager_ms = benchmark(lambda: model(x))
traced_ms = benchmark(lambda: traced(x))

print("\n=== EAGER VS TRACED ===")
print(f"Eager median: {eager_ms:.6f} ms/op")
print(f"Traced median: {traced_ms:.6f} ms/op")
print(f"Speed ratio (eager/traced): {eager_ms / traced_ms:.2f}x")

# Non-contiguous tensor benchmark
base = torch.randn(1024, 1024)
noncontiguous = base.t()
contiguous = noncontiguous.contiguous()

print("\n=== CONTIGUITY ===")
print("Contiguous:", contiguous.is_contiguous())
print("Non-contiguous:", noncontiguous.is_contiguous())
print("Shape:", noncontiguous.shape)
print("Dtype:", noncontiguous.dtype)

# Reduction along the same dimension, same numeric values
contiguous_ms = benchmark(
    lambda: contiguous.sum(dim=1), iterations=500
)
noncontiguous_ms = benchmark(
    lambda: noncontiguous.sum(dim=1), iterations=500
)

print("\n=== CONTIGUOUS VS NON-CONTIGUOUS ===")
print(f"Contiguous median: {contiguous_ms:.6f} ms/op")
print(f"Non-contiguous median: {noncontiguous_ms:.6f} ms/op")
print(f"Time ratio: {noncontiguous_ms / contiguous_ms:.2f}x")

with open(
    os.path.join(RESULTS_DIR, "framework_results.txt"),
    "w",
    encoding="utf-8"
) as f:
    f.write(f"Eager median ms/op: {eager_ms:.6f}\n")
    f.write(f"Traced median ms/op: {traced_ms:.6f}\n")
    f.write(f"Eager/traced ratio: {eager_ms / traced_ms:.2f}x\n")
    f.write(f"Contiguous median ms/op: {contiguous_ms:.6f}\n")
    f.write(f"Non-contiguous median ms/op: {noncontiguous_ms:.6f}\n")
    f.write(
        f"Non-contiguous/contiguous ratio: "
        f"{noncontiguous_ms / contiguous_ms:.2f}x\n"
    )

with open(
    os.path.join(RESULTS_DIR, "traced_graph.txt"),
    "w",
    encoding="utf-8"
) as f:
    f.write(str(traced.inlined_graph))

print("\nTASK 6 COMPLETED")
print("Results saved in results/")

