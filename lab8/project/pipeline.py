import random
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent

def build_loaders(cfg):

    transform = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])

    print("Loading GTSRB dataset...", flush=True)

    dataset = datasets.GTSRB(
        root=str(ROOT / "data"),
        split="train",
        download=True,
        transform=transform
    )

    test_dataset = datasets.GTSRB(
        root=str(ROOT / "data"),
        split="test",
        download=True,
        transform=transform
    )

    groups = defaultdict(list)

    for index, (path, label) in enumerate(dataset._samples):
        filename = Path(path).stem
        parts = filename.split("_")

        if len(parts) != 2:
            raise ValueError(f"Unexpected track filename: {filename}")

        track = parts[0]
        groups[(label, track)].append(index)

    rng = random.Random(cfg["seed"])

    train_indices = []
    val_indices = []

    for label in range(43):

        class_groups = [
            key for key in groups
            if key[0] == label
        ]

        rng.shuffle(class_groups)

        if len(class_groups) < 2:
            raise ValueError(f"Insufficient tracks for class {label}")

        n_val = max(
            1,
            round(len(class_groups) * cfg["val_fraction"])
        )

        n_val = min(n_val, len(class_groups) - 1)

        for key in class_groups[:n_val]:
            val_indices.extend(groups[key])

        for key in class_groups[n_val:]:
            train_indices.extend(groups[key])

    assert not set(train_indices).intersection(val_indices)

    counts = torch.zeros(43)

    for i in train_indices:
        counts[dataset._samples[i][1]] += 1

    weights = counts.sum() / (
        43 * counts.clamp_min(1)
    )

    print("Training images:", len(train_indices))
    print("Validation images:", len(val_indices))
    print("Test images:", len(test_dataset))
    print("Classes:", 43)

    options = {
        "batch_size": cfg["batch_size"],
        "num_workers": cfg["num_workers"]
    }

    return (
        DataLoader(
            Subset(dataset, train_indices),
            shuffle=True,
            **options
        ),
        DataLoader(
            Subset(dataset, val_indices),
            shuffle=False,
            **options
        ),
        DataLoader(
            test_dataset,
            shuffle=False,
            **options
        ),
        weights,
        transform
    )
