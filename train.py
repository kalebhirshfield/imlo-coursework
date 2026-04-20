import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F

transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

data = datasets.OxfordIIITPet(root="data", download=True, transform=transform)
segmentation_data = datasets.OxfordIIITPet(
    root="data", target_types="segmentation", transform=transform
)

for i in range(len(segmentation_data)):
    image, segmentation = segmentation_data[i]
    segmentation = transform(segmentation)
    mask = segmentation < 0.00784
    segmented_image = image * mask

train_data, val_data = random_split(data, [0.8, 0.2])
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=64)

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 61 * 61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 37)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(self.pool(F.relu(self.conv1(x))))))
        x = torch.flatten(x, 1)
        x = self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))
        return x


model = ImageClassifier().to(device)

lossFn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 30 * len(train_data) // 64

losses = []

for i in range(epochs):
    features, labels = next(iter(train_dataloader))
    features = features.to(device)
    labels = labels.to(device)
    pred = model(features)
    loss = lossFn(pred, labels)
    losses.append(loss.detach())

    if i % 10 == 0:
        print(f"Epoch {i} loss: {loss}")

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
