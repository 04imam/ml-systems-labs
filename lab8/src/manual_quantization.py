import torch
import json
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

checkpoint = ROOT.parent / "lab7" / "checkpoints" / "teacher.pt"

state = torch.load(
    checkpoint,
    map_location="cpu",
    weights_only=True
)

W = state["conv1.weight"].float()

print("Weight tensor:", W.shape)
print("Checkpoint loaded successfully")

def quantize(W, bits, per_channel=False):
    qmax = 2 ** (bits - 1) - 1

    if per_channel:
        maximum = W.abs().amax(
            dim=(1, 2, 3),
            keepdim=True
        )
    else:
        maximum = W.abs().max()

    scale = (maximum / qmax).clamp_min(1e-12)

    Q = torch.round(W / scale)
    Q = Q.clamp(-qmax, qmax).to(torch.int8)

    reconstructed = Q.float() * scale

    mse = ((W - reconstructed) ** 2).mean().item()
    mae = (W - reconstructed).abs().mean().item()

    return {
        "bits": bits,
        "method": "per_channel" if per_channel else "per_tensor",
        "mse": mse,
        "mae": mae,
        "scale_min": scale.min().item(),
        "scale_max": scale.max().item(),
        "zero_point": 0
    }

rows = []

for bits in [8, 4, 2]:
    for per_channel in [False, True]:
        result = quantize(W, bits, per_channel)
        rows.append(result)

        print(
            f"INT{bits} | {result['method']} | "
            f"MSE={result['mse']:.10f} | "
            f"MAE={result['mae']:.10f}"
        )

with open(RESULTS / "manual_quantization.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

with open(RESULTS / "manual_quantization.json", "w") as f:
    json.dump(rows, f, indent=2)

bits = [8, 4, 2]

for method in ["per_tensor", "per_channel"]:
    errors = [
        r["mse"] for r in rows
        if r["method"] == method
    ]
    plt.plot(bits, errors, marker="o", label=method)

plt.yscale("log")
plt.xticks(bits)
plt.gca().invert_xaxis()
plt.xlabel("Bit width")
plt.ylabel("Mean squared error")
plt.title("Manual Quantization Error")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS / "quantization_errors.png", dpi=200)
plt.close()

print("TASK 1 COMPLETED SUCCESSFULLY")
