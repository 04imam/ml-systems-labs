import sys
import copy
import csv
import json
import time
from pathlib import Path

import torch
import numpy as np
import yaml
from torch import nn
from torch.ao.quantization import quantize_dynamic

ROOT = Path(__file__).resolve().parents[1]
LAB7 = ROOT.parent / "lab7"

sys.path.insert(0, str(LAB7 / "src"))
from run_lab7 import CNN, loaders, seed_everything

RESULTS = ROOT / "results"
MODELS = ROOT / "models"

RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

cfg = yaml.safe_load(
    (LAB7 / "config.yaml").read_text()
)

seed_everything(cfg["seed"])
torch.set_num_threads(1)

engines = torch.backends.quantized.supported_engines

if "fbgemm" in engines:
    backend = "fbgemm"
elif "onednn" in engines:
    backend = "onednn"
else:
    raise RuntimeError("Required backend unavailable")

torch.backends.quantized.engine = backend
print("Backend:", backend)

model = CNN(
    cfg["teacher_channels"],
    cfg["teacher_hidden"]
)

model.load_state_dict(
    torch.load(
        LAB7 / "checkpoints" / "teacher.pt",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()
print("Teacher checkpoint loaded successfully")

_, _, test_loader = loaders(cfg)

def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for x, y in test_loader:
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100 * correct / total

def benchmark(model):
    x = next(iter(test_loader))[0][:1]
    times = []

    with torch.inference_mode():
        for _ in range(30):
            model(x)

        for _ in range(100):
            start = time.perf_counter()
            model(x)
            times.append(
                (time.perf_counter() - start) * 1000
            )

    return float(np.median(times)), float(np.percentile(times, 95))

def measure(model, name):
    print("Evaluating:", name, flush=True)

    accuracy = evaluate(model)
    median, p95 = benchmark(model)

    path = MODELS / f"{name}.pt"
    torch.save(model.state_dict(), path)

    result = {
        "model": name,
        "accuracy_pct": accuracy,
        "size_mb": path.stat().st_size / 1e6,
        "median_ms": median,
        "p95_ms": p95,
        "backend": backend,
        "cpu_threads": torch.get_num_threads()
    }

    print(json.dumps(result, indent=2))
    return result

baseline = measure(model, "baseline_fp32")

print("Applying dynamic INT8 quantization")

quantized = quantize_dynamic(
    copy.deepcopy(model),
    {nn.Linear},
    dtype=torch.qint8
)

quantized.eval()
print(quantized)

dynamic = measure(quantized, "dynamic_int8")

rows = [baseline, dynamic]

with open(RESULTS / "dynamic_quantization.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

with open(RESULTS / "dynamic_quantization.json", "w") as f:
    json.dump(rows, f, indent=2)

reduction = (1 - dynamic["size_mb"] / baseline["size_mb"]) * 100
speedup = baseline["median_ms"] / dynamic["median_ms"]
change = dynamic["accuracy_pct"] - baseline["accuracy_pct"]

print(f"Size reduction: {reduction:.2f}%")
print(f"Speedup: {speedup:.2f}x")
print(f"Accuracy change: {change:.2f} pp")

print("TASK 2 COMPLETED SUCCESSFULLY")
