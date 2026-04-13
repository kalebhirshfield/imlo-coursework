import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.OxfordIIITPet(root="data", download=True, transform=ToTensor())
