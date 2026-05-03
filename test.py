import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

data = datasets.OxfordIIITPet(
    root="data", split="test", download=True, transform=transform
)

test_dataloader = DataLoader(data, batch_size=64)

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


# Takes inspiration from: https://www.digitalocean.com/community/tutorials/writing-resnet-from-scratch-in-pytorch#resnet-from-scratch-how-models-work-in-pytorch
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)

        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        residual = self.skip(x)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.res1 = ResidualBlock(3, 32)
        self.res2 = ResidualBlock(32, 64)
        self.res3 = ResidualBlock(64, 128)
        self.res4 = ResidualBlock(128, 256)
        self.res5 = ResidualBlock(256, 256)
        self.res6 = ResidualBlock(256, 512)
        self.res7 = ResidualBlock(512, 512)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 256)
        self.bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 37)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.bn(self.fc1(x))))
        x = self.fc2(x)
        return x


model = ImageClassifier().to(device)
model.load_state_dict(torch.load("model.pth"))

correct = 0
total = 0

model.eval()
with torch.no_grad():
    for features, label in test_dataloader:
        features = features.to(device)
        label = label.to(device)
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

print(f"{100 * correct / total}% accuracy")
