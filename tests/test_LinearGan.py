import os
import pytest
import torch
import torchvision
from torchvision import transforms, datasets
from generateme.LinearGan import getLinearImage


def test_getLinearImage():
    image = getLinearImage()
    assert image.size == 4096
    
