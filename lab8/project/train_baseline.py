import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
import mlflow

from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import f1_score, classification_report
from PIL import Image

from pipeline import build_loaders

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"

CHECKPOINTS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

cfg = yaml.safe_load(
    (ROOT / "config.yaml").read_text()
)

SEED = cfg["seed"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(cfg["cpu_threads"])

device = torch.device("cpu")

print("Device:", device)
print("CPU threads:", torch.get_num_threads())

train_loader, val_loader, test_loader, weights, transform = (
    build_loaders(cfg)
)

latest_path = CHECKPOINTS / "latest.pt"
best_path = CHECKPOINTS / "baseline_best.pt"

pretrained = (
    ResNet18_Weights.DEFAULT
    if cfg["pretrained"] and not latest_path.exists()
    else None
)

model = resnet18(weights=pretrained)

model.fc = nn.Linear(
    model.fc.in_features,
    cfg["num_classes"]
)

model = model.to(device)

criterion = nn.CrossEntropyLoss(
    weight=weights.to(device)
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg["learning_rate"],
    weight_decay=cfg["weight_decay"]
)

start_epoch = 0
best_acc = -1.0
history = []
run_id = None

def atomic_save(obj, path):
    temporary = path.with_suffix(".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)

if latest_path.exists():

    saved = torch.load(
        latest_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])

    start_epoch = saved["epoch"]
    best_acc = saved["best_acc"]
    history = saved["history"]
    run_id = saved.get("run_id")

    print("Checkpoint loaded successfully")
    print("Completed epochs:", start_epoch)
    print("Best validation accuracy:", best_acc)

def evaluate(loader):

    model.eval()

    predictions = []
    targets = []

    with torch.inference_mode():

        for x, y in loader:

            output = model(x.to(device))
            pred = output.argmax(1)

            predictions.extend(pred.cpu().tolist())
            targets.extend(y.tolist())

    accuracy = 100 * np.mean(
        np.array(predictions) == np.array(targets)
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        labels=list(range(43)),
        average="macro",
        zero_division=0
    )

    return float(accuracy), float(macro_f1), targets, predictions

db_path = (ROOT / "mlflow.db").as_posix()

mlflow.set_tracking_uri("sqlite:///" + db_path)
mlflow.set_experiment("GTSRB_ResNet18_Baseline")

run_args = (
    {"run_id": run_id}
    if run_id
    else {"run_name": "resnet18_fp32"}
)

with mlflow.start_run(**run_args) as run:

    run_id = run.info.run_id

    mlflow.log_params(cfg)

    mlflow.log_params({
        "train_images": len(train_loader.dataset),
        "val_images": len(val_loader.dataset),
        "test_images": len(test_loader.dataset),
        "device": str(device)
    })

    for epoch in range(start_epoch, cfg["epochs"]):

        random.seed(SEED + epoch)
        np.random.seed(SEED + epoch)
        torch.manual_seed(SEED + epoch)

        model.train()

        loss_sum = 0.0
        correct = 0
        total = 0

        start = time.perf_counter()

        for batch, (x, y) in enumerate(train_loader, 1):

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = model(x)
            loss = criterion(output, y)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * y.size(0)

            correct += (
                output.argmax(1) == y
            ).sum().item()

            total += y.size(0)

            if batch % 50 == 0:

                print(
                    f"Epoch {epoch+1}/{cfg['epochs']} | "
                    f"Batch {batch}/{len(train_loader)} | "
                    f"Loss={loss_sum/total:.4f} | "
                    f"Acc={100*correct/total:.2f}%",
                    flush=True
                )

        train_acc = 100 * correct / total
        train_loss = loss_sum / total

        val_acc, val_f1, _, _ = evaluate(val_loader)

        elapsed = time.perf_counter() - start

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "epoch_seconds": elapsed
        }

        history.append(row)

        if val_acc > best_acc:

            best_acc = val_acc

            atomic_save(
                model.state_dict(),
                best_path
            )

            print("NEW BEST MODEL SAVED")

        atomic_save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc,
            "history": history,
            "run_id": run_id
        }, latest_path)

        (RESULTS / "training_history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8"
        )

        mlflow.log_metrics(
            {k: v for k, v in row.items() if k != "epoch"},
            step=epoch + 1
        )

        print(
            f"EPOCH {epoch+1}/{cfg['epochs']} COMPLETED | "
            f"Train={train_acc:.2f}% | "
            f"Val={val_acc:.2f}% | "
            f"F1={val_f1:.4f} | "
            f"Time={elapsed:.1f}s",
            flush=True
        )

    print("Loading best validation checkpoint")

    model.load_state_dict(
        torch.load(
            best_path,
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    print("Evaluating official GTSRB test set")

    test_acc, test_f1, targets, predictions = evaluate(
        test_loader
    )

    sample_paths = [
        path for path, _ in test_loader.dataset._samples[:100]
    ]

    samples = []

    for path in sample_paths:

        with Image.open(path) as img:
            samples.append(img.convert("RGB"))

    times = []

    with torch.inference_mode():

        for _ in range(30):

            x = transform(samples[0]).unsqueeze(0)
            model(x)

        for i in range(cfg["latency_runs"]):

            image = samples[i % len(samples)]

            start = time.perf_counter()

            x = transform(image).unsqueeze(0)
            model(x)

            times.append(
                (time.perf_counter() - start) * 1000
            )

    metrics = {
        "dataset": "GTSRB",
        "model": "ResNet18_FP32",
        "pretrained": cfg["pretrained"],
        "classes": 43,
        "image_size": cfg["image_size"],
        "train_images": len(train_loader.dataset),
        "validation_images": len(val_loader.dataset),
        "test_images": len(test_loader.dataset),
        "parameters": sum(
            p.numel() for p in model.parameters()
        ),
        "epochs": cfg["epochs"],
        "best_validation_accuracy_pct": best_acc,
        "test_accuracy_pct": test_acc,
        "test_macro_f1": test_f1,
        "model_size_mb": best_path.stat().st_size / 1e6,
        "median_latency_ms": float(np.median(times)),
        "p95_latency_ms": float(np.percentile(times, 95)),
        "latency_includes_preprocessing": True,
        "latency_includes_disk_io": False,
        "latency_runs": cfg["latency_runs"],
        "cpu_threads": cfg["cpu_threads"],
        "mlflow_run_id": run_id,
        "device": str(device)
    }

    (RESULTS / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )

    report = classification_report(
        targets,
        predictions,
        labels=list(range(43)),
        output_dict=True,
        zero_division=0
    )

    (RESULTS / "classification_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    mlflow.log_metrics({
        "test_accuracy_pct": test_acc,
        "test_macro_f1": test_f1,
        "model_size_mb": metrics["model_size_mb"],
        "median_latency_ms": metrics["median_latency_ms"],
        "p95_latency_ms": metrics["p95_latency_ms"]
    })

    mlflow.log_artifacts(str(RESULTS))
    mlflow.log_artifact(str(best_path))

    print(json.dumps(metrics, indent=2))

    print("TASK 6 BASELINE COMPLETED SUCCESSFULLY")
