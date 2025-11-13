"""
Script for training a traffic sign detection model.
"""

import torch
import torchvision
from lisa import LISA

if __name__ == "__main__":
    # load in LISA dataset
    dataset = LISA(root='data', download=True, train=True)