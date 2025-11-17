"""
Script for training a neural network to detect the bounding box of a traffic sign.
"""

import torch
import torchvision
import argparse
from parse_coco import parse_DFG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    trainloader, testloader = parse_DFG()