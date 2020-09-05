"""Simplified code which can be used for production purposes. Once the generator 
is trained we don't need the Discriminator so we can discard it."""
import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torchvision.utils import make_grid
from PIL import Image

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 64 * 64)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.fc4(x)
        
        return torch.tanh(x)

#create the generator and moving it to the gpu if available
G = Generator().to(device)

#loading the pretrained model
folder = os.path.dirname(os.path.abspath(__file__))
file = os.path.join(folder, "static/Models/linearGanGeneratorModel.pt")
G.load_state_dict(torch.load(file, map_location=torch.device("cpu")))

#You can call this method from an api for example and it will return the numpy
#corresponding to the image.
def getLinearImage():
    with torch.no_grad():
        test_z = torch.randn(1, 100).to(device)
        generated = G(test_z)
        generated = generated.cpu()
        grid = make_grid(generated[0].view(64, 64), normalize = True)
        im = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(im)
        return im