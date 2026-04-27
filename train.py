import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(256, (0.6, 1.0)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

data = datasets.OxfordIIITPet(root="data", download=True)

labels = [label for _, label in data]
train_indices, val_indices = train_test_split(
    range(len(data)), test_size=0.1, stratify=labels
)

train_data = datasets.OxfordIIITPet(root="data", transform=train_transform)
val_data = datasets.OxfordIIITPet(root="data", transform=val_transform)

train_subset = Subset(train_data, train_indices)
val_subset = Subset(val_data, val_indices)

train_dataloader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_subset, batch_size=64)

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 37)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.bn7(self.fc1(x))))
        x = self.fc2(x)
        return x


model = ImageClassifier().to(device)

lossFn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, factor=0.5, patience=2
)
epochs = 30

best_loss = float("inf")

for i in range(epochs):
    model.train()
    for features, label in train_dataloader:
        features = features.to(device)
        label = label.to(device)
        pred = model(features)
        loss = lossFn(pred, label)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for features, label in val_dataloader:
            features = features.to(device)
            label = label.to(device)
            outputs = model(features)
            val_loss += lossFn(outputs, label).item()
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    scheduler.step(val_loss)

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "model.pth")

    accuracy = correct * 100 / total

    print(f"Epoch {i+1} validation loss: {val_loss}, accuracy: {accuracy}%")
