import os
import pytest
import torch
import torchvision
from torchvision import transforms, datasets
from generateme.DCGan import getConvImage
    
def test_getConvImage():
    image = getConvImage()
    assert image.size == 12288

