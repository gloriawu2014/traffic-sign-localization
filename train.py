"""
Script for training a traffic sign detection model.
Dataset: https://www.vicos.si/resources/dfg/
Annotations are in the COCO json format compatible with Detectron/Mask-RCNN
"""

import torch
import torchvision
import torchvision.transforms as transforms
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
        self.root = image_folder
        self.transforms = transforms

        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        self.images = data["images"]
        self.annotations = data["annotations"]

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
            boxes.append(x, y, x+w, y+h)
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

"""
References: https://towardsdatascience.com/how-to-work-with-object-detection-datasets-in-coco-format-9bf4fb5848a4/
"""