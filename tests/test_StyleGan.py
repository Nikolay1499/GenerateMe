import os
import pytest
import torch
import torchvision
from torchvision import transforms, datasets
from generateme.StyleGan import getStyleImage


def test_getStyleImage():
    image = getStyleImage()
    assert image.size == 12288