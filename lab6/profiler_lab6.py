import time
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity, record_function
from models import CNN
from dataset import train_dataset

torch.manual_seed(42)
torch.set_num_threads(2)

loader = DataLoader(
    train_dataset, batch_size=128,
    shuffle=False, num_workers=0
)

model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
timings = {"data_loading": 0, "forward": 0,
           "backward": 0, "optimizer": 0}

iterator = iter(loader)

with profile(activities=[ProfilerActivity.CPU]) as prof:
    while True:
        start = time.perf_counter()

        with record_function("data_loading"):
            try:
                images, labels = next(iterator)
            except StopIteration:
                break

        timings["data_loading"] += time.perf_counter() - start

        start = time.perf_counter()
        with record_function("forward"):
            output = model(images)
            loss = criterion(output, labels)
        timings["forward"] += time.perf_counter() - start

        optimizer.zero_grad()
        start = time.perf_counter()
        with record_function("backward"):
            loss.backward()
        timings["backward"] += time.perf_counter() - start

        start = time.perf_counter()
        with record_function("optimizer"):
            optimizer.step()
        timings["optimizer"] += time.perf_counter() - start

total = sum(timings.values())
results = {
    k: {
        "seconds": round(v, 4),
        "percent": round(100*v/total, 2)
    }
    for k, v in timings.items()
}

import os
os.makedirs("results", exist_ok=True)

with open("results/profiling.json", "w") as f:
    json.dump(results, f, indent=2)

with open("results/profiler_table.txt", "w") as f:
    f.write(prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=20
    ))

for k, v in results.items():
    print(f"{k}: {v['percent']}%")

print("TASK 2 COMPLETED")
