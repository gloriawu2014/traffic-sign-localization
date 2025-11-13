"""
Script for training a traffic sign detection model using YOLO.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
from lisa import LISA


def data():
    # load and process LISA dataset
    LISA(root='data', download=True, train=True)
    images_tensor_0 = torch.load("data/lisa-batches/images_0.tensor")
    images_tensor_1 = torch.load("data/lisa-batches/images_1.tensor")
    images_tensor_2 = torch.load("data/lisa-batches/images_2.tensor")
    labels_tensor = torch.load("data/lisa-batches/labels.tensor")
    #print(images_tensor_0.shape) [2619, 3, 32, 32]
    #print(images_tensor_1.shape) [2619, 3, 32, 32]
    #print(images_tensor_2.shape) [2619, 3, 32, 32]
    #print(labels_tensor.shape)  [7855]
    #print(labels_tensor[0]) tensor(0)

    return images_tensor_0, images_tensor_1, images_tensor_2, labels_tensor

    ### classification IDs, not info about bounding box

def view_images(images_tensor):
    # convert tensor to images
    output_dir = "data/lisa_images/images_2"
    os.makedirs(output_dir, exist_ok=True)
    
    to_pil = transforms.ToPILImage()

    for i, img_tensor in enumerate(images_tensor):
        img = to_pil(img_tensor)
        img.save(os.path.join(output_dir, f"{i:05d}.png"))

if __name__ == "__main__":
    # load in LISA dataset
    images_0, images_1, images_2, labels = data()
    #view_images(images_2)