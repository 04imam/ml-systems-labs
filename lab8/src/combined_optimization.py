import csv
import json
import copy

import torch
from torch import nn
from torch.ao.quantization import quantize_dynamic

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from static_quantization import (
    load_model,
    static_quantize,
    measure,
    RESULTS,
    backend
)

torch.set_num_threads(1)

def main():

    teacher = load_model("teacher")
    pruned = load_model("structured_0p5")
    distilled = load_model("student_distilled")

    experiments = [
        ("baseline_fp32", teacher),
        ("pruned_fp32", pruned),
        ("distilled_fp32", distilled),
        (
            "dynamic_int8",
            lambda: quantize_dynamic(
                copy.deepcopy(teacher),
                {nn.Linear},
                dtype=torch.qint8
            ).eval()
        ),
        (
            "static_int8",
            lambda: static_quantize(teacher)
        ),
        (
            "pruned_static_int8",
            lambda: static_quantize(pruned)
        ),
        (
            "distilled_static_int8",
            lambda: static_quantize(distilled)
        )
    ]

    result_path = RESULTS / "variant_comparison.json"
    csv_path = RESULTS / "variant_comparison.csv"

    rows = []

    if result_path.exists():
        rows = json.loads(
            result_path.read_text(encoding="utf-8")
        )

    completed = {r["model"] for r in rows}

    for index, (name, factory) in enumerate(experiments, 1):

        if name in completed:
            print("SKIPPING:", name)
            continue

        print("\n" + "=" * 50)
        print(f"EXPERIMENT {index}/7: {name}")
        print("=" * 50)

        model = (
            factory
            if isinstance(factory, nn.Module)
            else factory()
        )

        model.eval()

        result = measure(model, name)

        result["calibration_images"] = (
            500 if "static" in name else 0
        )

        rows.append(result)

        result_path.write_text(
            json.dumps(rows, indent=2),
            encoding="utf-8"
        )

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=rows[0].keys()
            )
            writer.writeheader()
            writer.writerows(rows)

        print("SAVED:", name, flush=True)

    print("\nFINAL MODEL COMPARISON")

    for r in rows:
        print(
            f"{r['model']:25s} "
            f"Accuracy={r['accuracy_pct']:.2f}% | "
            f"Size={r['size_mb']:.3f} MB | "
            f"Latency={r['median_ms']:.3f} ms"
        )

    labels = [r["model"] for r in rows]
    accuracy = [r["accuracy_pct"] for r in rows]
    latency = [r["median_ms"] for r in rows]
    sizes = [r["size_mb"] for r in rows]

    plt.figure(figsize=(11, 7))

    plt.scatter(latency, accuracy, s=90)

    for x, y, label in zip(
        latency, accuracy, labels
    ):
        plt.annotate(
            label,
            (x, y),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8
        )

    plt.xlabel("Median CPU latency (ms)")
    plt.ylabel("Test accuracy (%)")
    plt.title("Accuracy vs Inference Latency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        RESULTS / "accuracy_vs_latency.png",
        dpi=200
    )
    plt.close()

    plt.figure(figsize=(11, 7))

    plt.scatter(sizes, accuracy, s=90)

    for x, y, label in zip(
        sizes, accuracy, labels
    ):
        plt.annotate(
            label,
            (x, y),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8
        )

    plt.xlabel("Model size (MB)")
    plt.ylabel("Test accuracy (%)")
    plt.title("Accuracy vs Model Size")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        RESULTS / "accuracy_vs_size.png",
        dpi=200
    )
    plt.close()

    print("\nTASK 5 EXPERIMENTS COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()
