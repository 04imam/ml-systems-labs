
import json
import statistics
import time

import torch
from thop import profile
from torchinfo import summary

from models import MLP, CNN, AttentionModel

torch.set_num_threads(4)

results = {}

for name, Model in [
    ("MLP", MLP),
    ("CNN", CNN),
    ("Attention", AttentionModel)
]:
    model = Model().cpu().eval()
    model.load_state_dict(
        torch.load(
            f"results/{name.lower()}.pt",
            map_location="cpu",
            weights_only=True
        )
    )

    x = torch.randn(1, 1, 28, 28)
    print(f"\n{name}", flush=True)

    info = summary(model, input_size=(1, 1, 28, 28),
                   verbose=0)
    parameters = info.total_params

    macs, _ = profile(model, inputs=(x,), verbose=False)
    estimated_flops = 2 * macs

    latencies = {}

    for batch_size, runs in [(1, 100), (128, 20)]:
        batch = torch.randn(batch_size, 1, 28, 28)

        with torch.inference_mode():
            for _ in range(5):
                model(batch)

            times = []
            for _ in range(runs):
                start = time.perf_counter()
                model(batch)
                times.append((time.perf_counter() - start) * 1000)

        latencies[str(batch_size)] = statistics.median(times)

    results[name] = {
        "parameters": parameters,
        "thop_macs": macs,
        "estimated_flops": estimated_flops,
        "latency_batch_1_ms": latencies["1"],
        "latency_batch_128_ms": latencies["128"]
    }

    print(f"Parameters: {parameters:,}")
    print(f"THOP MACs: {macs:,.0f}")
    print(f"Estimated FLOPs: {estimated_flops:,.0f}")
    print(f"Batch 1 latency: {latencies['1']:.3f} ms")
    print(f"Batch 128 latency: {latencies['128']:.3f} ms")

with open("results/profile_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nProfiling complete!")
