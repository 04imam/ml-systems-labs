
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        return self.net(x)


class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch = nn.Linear(49, 64)

        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        patches = x.unfold(2, 7, 7).unfold(3, 7, 7)
        patches = patches.contiguous().view(x.size(0), 16, 49)

        x = self.patch(patches)
        x = self.encoder(x)
        return self.classifier(x.mean(dim=1))


if __name__ == "__main__":
    sample = torch.randn(2, 1, 28, 28)

    for name, model in [
        ("MLP", MLP()),
        ("CNN", CNN()),
        ("Attention", AttentionModel())
    ]:
        output = model(sample)
        params = sum(p.numel() for p in model.parameters())

        print(f"{name}: output={tuple(output.shape)}, parameters={params:,}")
