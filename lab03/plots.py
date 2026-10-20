
import json
import os
import matplotlib.pyplot as plt

with open("results/training_results.json") as f:
    training = json.load(f)

with open("results/profile_results.json") as f:
    profiling = json.load(f)

os.makedirs("results", exist_ok=True)

models = ["MLP", "CNN", "Attention"]
accuracy = [training[m]["test_accuracy"] for m in models]
parameters = [profiling[m]["parameters"] for m in models]
latency = [profiling[m]["latency_batch_1_ms"] for m in models]

# Graph 1: Accuracy vs Parameters
plt.figure(figsize=(8, 5))
for i, model in enumerate(models):
    plt.scatter(parameters[i], accuracy[i], s=120)
    plt.annotate(model, (parameters[i], accuracy[i]),
                 xytext=(8, 5), textcoords="offset points")

plt.xlabel("Number of Parameters")
plt.ylabel("Test Accuracy (%)")
plt.title("Fashion-MNIST: Accuracy vs Parameters")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/accuracy_vs_parameters.png", dpi=200)
plt.close()

# Graph 2: Accuracy vs Latency
plt.figure(figsize=(8, 5))
for i, model in enumerate(models):
    plt.scatter(latency[i], accuracy[i], s=120)
    plt.annotate(model, (latency[i], accuracy[i]),
                 xytext=(8, 5), textcoords="offset points")

plt.xlabel("Median Inference Latency - Batch 1 (ms)")
plt.ylabel("Test Accuracy (%)")
plt.title("Fashion-MNIST: Accuracy vs Latency")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/accuracy_vs_latency.png", dpi=200)
plt.close()

print("Both graphs generated successfully!")
