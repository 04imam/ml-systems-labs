import os
import random
import argparse
import numpy as np
import torch
import yaml
import mlflow
from torch import nn
from torch.utils.data import DataLoader
from models import CNN
from dataset import train_dataset, val_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/config.yaml")
parser.add_argument("--resume", default=None)
args = parser.parse_args()

with open(args.config, encoding="utf-8-sig") as f:
    cfg = yaml.safe_load(f)

seed = cfg["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

training = cfg["training"]
batch_size = training["batch_size"]
epochs = training["epochs"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=training["num_workers"],
    generator=torch.Generator().manual_seed(seed)
)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=training["learning_rate"]
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=3, gamma=0.5
)

os.makedirs(cfg["checkpoint"]["directory"], exist_ok=True)
mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)
mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

start_epoch = 1
if args.resume:
    checkpoint = torch.load(
        args.resume, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming from epoch {start_epoch}")

with mlflow.start_run():
    mlflow.log_params({
        "seed": seed,
        "model": cfg["model"],
        "dataset": cfg["dataset"],
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer": training["optimizer"],
        "scheduler": training["scheduler"],
        "learning_rate": training["learning_rate"],
        "num_workers": training["num_workers"]
    })

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0

        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                predictions = model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        scheduler.step()

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Loss: {total_loss / len(train_loader):.4f} | "
            f"Validation: {accuracy:.2f}%",
            flush=True
        )

        mlflow.log_metric("train_loss", total_loss / len(train_loader), step=epoch)
        mlflow.log_metric("val_accuracy", accuracy, step=epoch)

        checkpoint_path = os.path.join(
            cfg["checkpoint"]["directory"],
            f"cnn_epoch_{epoch}.pt"
        )
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        }, checkpoint_path)

    mlflow.log_artifact(args.config)
    mlflow.log_artifact(checkpoint_path)

print("TRAINING COMPLETE")

