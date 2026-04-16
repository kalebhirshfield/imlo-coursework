import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

data = datasets.OxfordIIITPet(root="data", download=True, transform=transform)
train_data, val_data = random_split(data, [0.7, 0.3])
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=64)

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
