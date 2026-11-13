import os
import time
import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from torch import nn
from torch.utils.data import DataLoader
from models import CNN
from dataset import train_dataset, val_dataset

def main():
    os.makedirs("results", exist_ok=True)
    torch.set_num_threads(2)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Lab6_Optimizer_Scheduler_Study")

    rows = []
    history = []

    for optimizer_name in ["Adam", "SGD"]:
        for scheduler_name in [
            "StepLR",
            "CosineAnnealingLR",
            "ReduceLROnPlateau"
        ]:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)

            model = CNN()
            criterion = nn.CrossEntropyLoss()

            train_loader = DataLoader(
                train_dataset,
                batch_size=64,
                shuffle=True,
                num_workers=0,
                generator=torch.Generator().manual_seed(42)
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=128,
                num_workers=0
            )

            lr = 0.001 if optimizer_name == "Adam" else 0.01

            if optimizer_name == "Adam":
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=lr
                )
            else:
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=lr,
                    momentum=0.9
                )

            if scheduler_name == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=3, gamma=0.5
                )
            elif scheduler_name == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=10
                )
            else:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min",
                    factor=0.5, patience=2
                )

            name = optimizer_name + "_" + scheduler_name
            start = time.perf_counter()

            with mlflow.start_run(run_name=name):
                mlflow.log_params({
                    "optimizer": optimizer_name,
                    "scheduler": scheduler_name,
                    "initial_lr": lr,
                    "epochs": 10,
                    "batch_size": 64,
                    "seed": 42
                })

                final_accuracy = 0
                best_accuracy = 0

                for epoch in range(1, 11):
                    model.train()
                    running_loss = 0

                    for images, labels in train_loader:
                        optimizer.zero_grad()
                        output = model(images)
                        loss = criterion(output, labels)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()

                    model.eval()
                    correct = 0
                    total = 0
                    val_loss_sum = 0

                    with torch.inference_mode():
                        for images, labels in val_loader:
                            output = model(images)
                            loss = criterion(output, labels)
                            val_loss_sum += loss.item() * len(labels)
                            correct += (
                                output.argmax(1) == labels
                            ).sum().item()
                            total += len(labels)

                    final_accuracy = 100 * correct / total
                    best_accuracy = max(
                        best_accuracy, final_accuracy
                    )
                    val_loss = val_loss_sum / total
                    current_lr = optimizer.param_groups[0]["lr"]

                    mlflow.log_metrics({
                        "validation_accuracy": final_accuracy,
                        "validation_loss": val_loss,
                        "train_loss": running_loss / len(train_loader),
                        "learning_rate": current_lr
                    }, step=epoch)

                    history.append({
                        "experiment": name,
                        "epoch": epoch,
                        "accuracy": final_accuracy,
                        "learning_rate": current_lr
                    })

                    if scheduler_name == "ReduceLROnPlateau":
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

                    print(
                        f"{name} | Epoch {epoch}/10 | "
                        f"Accuracy {final_accuracy:.2f}%",
                        flush=True
                    )

                duration = time.perf_counter() - start

                mlflow.log_metrics({
                    "final_accuracy": final_accuracy,
                    "best_accuracy": best_accuracy,
                    "training_seconds": duration
                })

            rows.append({
                "optimizer": optimizer_name,
                "scheduler": scheduler_name,
                "final_accuracy": round(final_accuracy, 2),
                "best_accuracy": round(best_accuracy, 2),
                "training_seconds": round(duration, 2)
            })

            pd.DataFrame(rows).to_csv(
                "results/optimizer_scheduler_study.csv",
                index=False
            )
            pd.DataFrame(history).to_csv(
                "results/optimizer_scheduler_history.csv",
                index=False
            )

            print(f"COMPLETED: {name}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("results/optimizer_scheduler_study.csv", index=False)

    plt.figure(figsize=(10, 6))
    for name, group in pd.DataFrame(history).groupby("experiment"):
        plt.plot(
            group["epoch"], group["accuracy"],
            marker="o", label=name
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Lab 6 - Optimizer and Scheduler Comparison")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/optimizer_scheduler_comparison.png")
    plt.close()

    print("\nFINAL RESULTS")
    print(df.to_string(index=False))
    print("\nTASK 5 COMPLETED")

if __name__ == "__main__":
    main()
