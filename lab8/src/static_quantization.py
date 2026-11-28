import sys
import copy
import csv
import json
import time
from pathlib import Path

import torch
import numpy as np
import yaml

from torch.utils.data import DataLoader, Subset
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

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

def load_model(name):
    state = torch.load(
        LAB7 / "checkpoints" / f"{name}.pt",
        map_location="cpu",
        weights_only=True
    )

    channels = [
        state["conv1.weight"].shape[0],
        state["conv2.weight"].shape[0],
        state["conv3.weight"].shape[0]
    ]

    hidden = state["fc1.weight"].shape[0]

    model = CNN(channels, hidden)
    model.load_state_dict(state)

    return model.cpu().eval()

train_loader, _, test_loader = loaders(cfg)

generator = torch.Generator().manual_seed(cfg["seed"])

indices = torch.randperm(
    len(train_loader.dataset),
    generator=generator
)[:500].tolist()

calibration_loader = DataLoader(
    Subset(train_loader.dataset, indices),
    batch_size=50,
    shuffle=False,
    num_workers=0
)

def static_quantize(model):
    mapping = get_default_qconfig_mapping(backend)

    example = (torch.randn(1, 1, 28, 28),)

    prepared = prepare_fx(
        copy.deepcopy(model).eval(),
        mapping,
        example
    )

    print("Calibrating on 500 training images")

    count = 0

    with torch.no_grad():
        for x, _ in calibration_loader:
            prepared(x)
            count += x.size(0)
            print(f"Calibration: {count}/500", flush=True)

    assert count == 500

    quantized = convert_fx(prepared).eval()

    quantized_layers = {
        name: type(module).__name__
        for name, module in quantized.named_modules()
        if "quantized" in type(module).__module__
        and "Conv" in type(module).__name__
        or "quantized" in type(module).__module__
        and "Linear" in type(module).__name__
    }

    print("Quantized layers:", quantized_layers)

    if not any("Conv" in v for v in quantized_layers.values()):
        raise RuntimeError("Convolutions were not quantized")

    if not any("Linear" in v for v in quantized_layers.values()):
        raise RuntimeError("Linear layers were not quantized")

    return quantized

def evaluate(model):
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

    return (
        float(np.median(times)),
        float(np.percentile(times, 95))
    )

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
        "cpu_threads": torch.get_num_threads(),
        "calibration_images": 500
    }

    print(json.dumps(result, indent=2))
    return result

def main():
    teacher = load_model("teacher")

    print("Teacher checkpoint loaded successfully")

    quantized = static_quantize(teacher)

    result = measure(quantized, "static_int8")

    with open(
        RESULTS / "static_quantization.csv",
        "w",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=result.keys()
        )
        writer.writeheader()
        writer.writerow(result)

    with open(
        RESULTS / "static_quantization.json",
        "w"
    ) as f:
        json.dump(result, f, indent=2)

    print("TASK 3 COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()
