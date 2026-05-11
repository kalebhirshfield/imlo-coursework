import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

train_transform = transforms.Compose(
    [
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, (0.6, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

data = datasets.OxfordIIITPet(root="data", download=True, transform=train_transform)

train_dataloader = DataLoader(data, batch_size=64, shuffle=True)

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
        x = self.pool(F.silu(self.bn1(self.conv1(x))))
        x = self.bn2(self.conv2(x))
        x = F.silu(x + residual)
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
        x = self.dropout(F.silu(self.bn(self.fc1(x))))
        x = self.fc2(x)
        return x


model = ImageClassifier().to(device)

lossFn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, 30)
epochs = 30

for i in range(epochs):
    model.train()
    correct = 0
    total = 0
    train_loss = 0
    for features, label in train_dataloader:
        features = features.to(device)
        label = label.to(device)
        outputs = model(features)
        loss = lossFn(outputs, label)
        train_loss += loss.item() * label.size(0)
        _, predicted = torch.max(outputs, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    train_loss /= total
    train_accuracy = correct * 100 / total
    lr = optimiser.param_groups[0]["lr"]

    print(
        f"Epoch {i+1}, train loss: {train_loss}, train accuracy: {train_accuracy}%, learning rate: {lr}"
    )

    scheduler.step()

model.eval()
with torch.no_grad():
    for features, label in train_dataloader:
        features = features.to(device)
        label = label.to(device)
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

print(f"{100 * correct / total}% accuracy")

torch.save(model.state_dict(), "model.pth")
