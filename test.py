import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

data = datasets.OxfordIIITPet(
    root="data", split="test", download=True, transform=transform
)
segmentation_data = datasets.OxfordIIITPet(
    root="data", split="test", target_types="segmentation", transform=transform
)

segmented_data = []
for i in range(len(segmentation_data)):
    image, segmentation = segmentation_data[i]
    segmentation = transform(segmentation)
    mask = (segmentation > 0.01176) | (segmentation < 0.00784)
    segmented_image = image * mask
    segmented_data.append((segmented_image, data[i][1]))

test_dataloader = DataLoader(segmented_data, batch_size=64)

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.batchnorm
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 30 * 30, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 37)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


model = ImageClassifier().to(device)
model.load_state_dict(torch.load("model.pth"))

lossFn = nn.CrossEntropyLoss()

correct = 0
total = 0

with torch.no_grad():
    for features, label in test_dataloader:
        features = features.to(device)
        label = label.to(device)
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

print(f"{100 * correct / total}% accuracy")
