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
from torchvision.utils import save_image

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
G.load_state_dict(torch.load("static/Models/linearGanGeneratorModel.pt", map_location=torch.device("cpu")))

#You can call this method from an api for example and it will return the numpy
#corresponding to the image.
def getLinearImage():
  with torch.no_grad():
      test_z = torch.randn(1, 100).to(device)
      generated = G(test_z)
      generated = generated.cpu()
      numpy = generated[0].view(64, 64).numpy()
      numpy = numpy / 2 + 0.5
      save_image(generated[0].view(64, 64), "static/Photos/image.png", normalize = True)
      return numpy