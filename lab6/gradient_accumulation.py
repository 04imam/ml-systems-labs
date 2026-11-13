import os
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from models import CNN
from dataset import train_dataset, val_dataset

def run_experiment(name, batch_size, accumulation):
    torch.manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(42)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=128, num_workers=0
    )

    model = CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    history = []
    start = time.perf_counter()

    for epoch in range(1, 6):
        model.train()
        optimizer.zero_grad()
        running_loss = 0
        total_samples = 0

        for index, (images, labels) in enumerate(train_loader):
            output = model(images)
            loss = criterion(output, labels)

            group_start = (index // accumulation) * batch_size * accumulation
            group_size = min(
                batch_size * accumulation,
                len(train_dataset) - group_start
            )

            (loss * len(labels) / group_size).backward()

            running_loss += loss.item() * len(labels)
            total_samples += len(labels)

            if (index + 1) % accumulation == 0 or index + 1 == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        correct = 0
        total = 0

        with torch.inference_mode():
            for images, labels in val_loader:
                predictions = model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += len(labels)

        accuracy = 100 * correct / total

        history.append({
            "experiment": name,
            "epoch": epoch,
            "train_loss": running_loss / total_samples,
            "validation_accuracy": accuracy
        })

        print(
            f"{name} | Epoch {epoch}/5 | Accuracy {accuracy:.2f}%",
            flush=True
        )

    print(f"{name} time: {time.perf_counter()-start:.2f}s")
    return history

def main():
    os.makedirs("results", exist_ok=True)
    torch.set_num_threads(2)

    normal = run_experiment("Normal_Batch256", 256, 1)
    accumulated = run_experiment("Accumulation_64x4", 64, 4)

    df = pd.DataFrame(normal + accumulated)
    df.to_csv("results/gradient_accumulation.csv", index=False)

    plt.figure(figsize=(8, 5))
    for name, group in df.groupby("experiment"):
        plt.plot(
            group["epoch"],
            group["validation_accuracy"],
            marker="o",
            label=name
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Normal Batch vs Gradient Accumulation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/gradient_accumulation.png")
    plt.close()

    difference = abs(
        normal[-1]["validation_accuracy"] -
        accumulated[-1]["validation_accuracy"]
    )

    print(f"Final accuracy difference: {difference:.2f} percentage points")
    print("TASK 6 COMPLETED")

if __name__ == "__main__":
    main()
