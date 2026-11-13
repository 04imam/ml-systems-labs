import os
import time
import threading
import psutil
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from models import CNN
from dataset import train_dataset, val_dataset

def main():
    os.makedirs("results", exist_ok=True)
    torch.set_num_threads(2)
    process = psutil.Process()
    rows = []

    for batch_size in [32, 64, 128, 256, 512]:
        torch.manual_seed(42)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(42)
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            num_workers=0
        )

        model = CNN()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.001
        )
        criterion = nn.CrossEntropyLoss()

        peak = [process.memory_info().rss]
        stop = threading.Event()

        def monitor():
            while not stop.wait(0.02):
                peak[0] = max(
                    peak[0], process.memory_info().rss
                )

        watcher = threading.Thread(target=monitor)
        watcher.start()
        start = time.perf_counter()

        for epoch in range(5):
            model.train()
            for images, labels in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()

        seconds = time.perf_counter() - start
        stop.set()
        watcher.join()
        peak[0] = max(peak[0], process.memory_info().rss)

        model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                predictions = model(images).argmax(1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        throughput = len(train_dataset) * 5 / seconds
        memory = peak[0] / (1024 ** 2)

        rows.append({
            "batch_size": batch_size,
            "throughput_samples_per_sec": round(throughput, 2),
            "peak_process_memory_mb": round(memory, 2),
            "validation_accuracy_percent": round(accuracy, 2)
        })

        print(
            f"Batch {batch_size} | "
            f"Throughput {throughput:.2f} samples/s | "
            f"Peak RAM {memory:.2f} MB | "
            f"Accuracy {accuracy:.2f}%",
            flush=True
        )

    df = pd.DataFrame(rows)
    df.to_csv("results/batch_study.csv", index=False)

    for column, title, filename in [
        ("throughput_samples_per_sec", "Throughput", "batch_throughput.png"),
        ("peak_process_memory_mb", "Peak Process RAM", "batch_memory.png"),
        ("validation_accuracy_percent", "Validation Accuracy", "batch_accuracy.png")
    ]:
        plt.figure(figsize=(8, 5))
        plt.plot(df["batch_size"], df[column], marker="o")
        plt.xlabel("Batch Size")
        plt.ylabel(title)
        plt.title("Lab 6 - " + title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("results/" + filename)
        plt.close()

    print("TASK 4 COMPLETED")

if __name__ == "__main__":
    main()
