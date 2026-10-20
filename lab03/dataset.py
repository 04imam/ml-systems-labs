
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_full = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_dataset, val_dataset = random_split(
    train_full,
    [55000, 5000],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True
)

val_loader = DataLoader(
    val_dataset, batch_size=128
)

test_loader = DataLoader(
    test_dataset, batch_size=128
)

if __name__ == "__main__":
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Testing images:", len(test_dataset))
    print("Batch size:", 128)
