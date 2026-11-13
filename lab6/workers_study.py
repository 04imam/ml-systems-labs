import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import train_dataset

def main():
    os.makedirs("results", exist_ok=True)
    rows = []

    for workers in [0, 1, 2, 4, 8]:
        loader = DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=workers
        )

        start = time.perf_counter()
        samples = 0

        for images, labels in loader:
            samples += len(labels)

        seconds = time.perf_counter() - start
        throughput = samples / seconds

        rows.append({
            "num_workers": workers,
            "seconds": round(seconds, 3),
            "samples_per_second": round(throughput, 2)
        })

        print(
            f"Workers: {workers} | "
            f"Time: {seconds:.2f}s | "
            f"Throughput: {throughput:.2f} samples/s",
            flush=True
        )

    df = pd.DataFrame(rows)
    df.to_csv("results/workers_study.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(
        df["num_workers"],
        df["samples_per_second"],
        marker="o"
    )
    plt.xlabel("Number of DataLoader Workers")
    plt.ylabel("Throughput (samples/second)")
    plt.title("Lab 6 - DataLoader Worker Comparison")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/workers_study.png")
    plt.close()

    print("TASK 3 COMPLETED")

if __name__ == "__main__":
    main()
