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

segmented_data = []
for i in range(len(segmentation_data)):
    image, segmentation = segmentation_data[i]
    segmentation = transform(segmentation)
    mask = (segmentation > 0.01176) | (segmentation < 0.00784)
    segmented_image = image * mask
    # plt.imshow(segmented_image.squeeze().permute(1, 2, 0))
    # plt.show()
    segmented_data.append((segmented_image, data[i][1]))

segmented_dataloader = DataLoader(segmented_data, batch_size=64, shuffle=True)

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
epochs = 30

for i in range(epochs):
    epoch_loss = 0
    for features, label in segmented_dataloader:
        features = features.to(device)
        label = label.to(device)
        pred = model(features)
        loss = lossFn(pred, label)
        epoch_loss += loss

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"Epoch {i+1} loss: {epoch_loss}")


torch.save(model.state_dict(), "model.pth")
