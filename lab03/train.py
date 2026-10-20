
import os
import time
import json
import torch
from torch import nn
from models import MLP, CNN, AttentionModel
from dataset import train_loader, val_loader, test_loader

torch.manual_seed(42)
torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("results", exist_ok=True)

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

results = {}

for name, model in [
    ("MLP", MLP()),
    ("CNN", CNN()),
    ("Attention", AttentionModel())
]:
    print(f"\nTraining {name} on {device}", flush=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    start = time.perf_counter()

    for epoch in range(1, 6):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_acc = evaluate(model, val_loader)
        print(
            f"{name} Epoch {epoch}/5 | "
            f"Loss: {running_loss/len(train_loader):.4f} | "
            f"Validation: {val_acc:.2f}%",
            flush=True
        )

    test_acc = evaluate(model, test_loader)
    duration = time.perf_counter() - start

    results[name] = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "test_accuracy": test_acc,
        "training_seconds": duration
    }

    torch.save(model.state_dict(), f"results/{name.lower()}.pt")
    print(f"{name} TEST ACCURACY: {test_acc:.2f}%", flush=True)

with open("results/training_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nTraining complete!", flush=True)
