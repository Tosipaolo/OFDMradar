import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from OFDMradar import OFDMradar
from Target import Target
from TargetDetection import TargetDetector
from plotStatistics import *
import time

# tensorflow:
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class myDenseLayer():

