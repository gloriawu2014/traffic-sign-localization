"""
Script for parsing annotations in COCO json format
Dataset: https://www.vicos.si/resources/dfg/
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json
import os
from PIL import Image

"""
Class for parsing annotations in COCO json format.
Requires a folder with all images and a json file with annotations.
"""
class COCOTrafficSigns(torch.utils.data.Dataset):
    def __init__(self, image_folder, annotation_file, transforms=None):
        """
        image_folder: folder containing all images
        annotation_file: path to single JSON file
        """
        if not os.path.isdir(image_folder):
            raise FileNotFoundError(
                f"[ERROR] Image folder not found: '{image_folder}'"
            )
        
        if not os.path.isfile(annotation_file):
            raise FileNotFoundError(
                f"[ERROR] Annotation file not found: '{annotation_file}"
            )
        
        self.root = image_folder
        self.transforms = transforms

        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        self.images = data["images"]
        self.annotations = data["annotations"]

        missing_images = []
        for img in self.images:
            path = os.path.join(image_folder, img["file_name"])
            if not os.path.isfile(path):
                missing_images.append(img["file_name"])

        if missing_images:
            raise FileNotFoundError(
                f"[ERROR] The following images listed in the annotation file do not exist:\n"
                + "\n".join(missing_images[:10]) +
                ("\n... (more missing)" if len(missing_images) > 10 else "")
            )

        # dictionary: img_id -> list of annotations
        self.ann_map = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.ann_map:
                self.ann_map[img_id] = []
            self.ann_map[img_id].append(ann)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.root, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        img_id = img_info["id"]
        anns = self.ann_map.get(img_id, [])

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x+w, y+h])
            labels.append(ann["category_id"]) # = 1 for traffic sign

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target
    
def collate_fn(batch):
    return tuple(zip(*batch))

def parse_DFG():
    testset = COCOTrafficSigns(
        image_folder="data/JPEGImages",
        annotation_file="data/DFG-tsd-annot-json/test.json",
        transforms=transforms.ToTensor()
    )
    trainset = COCOTrafficSigns(
        image_folder="data/JPEGImages",
        annotation_file="data/DFG-tsd-annot-json/train.json",
        transforms=transforms.ToTensor()
    )

    testloader = DataLoader(testset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    return trainloader, testloader

"""
References: https://towardsdatascience.com/how-to-work-with-object-detection-datasets-in-coco-format-9bf4fb5848a4/
"""