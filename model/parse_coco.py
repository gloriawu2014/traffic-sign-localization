"""
Script for parsing annotations in COCO json format
Dataset: https://www.vicos.si/resources/dfg/
"""

import torch
import torchvision
import torchvision.transforms.functional as F_t
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from pycocotools import mask as mask_utils
import numpy as np
import json
import os
from PIL import Image

"""
Class for resizing images for faster computation
"""


class ResizeTransform:
    def __init__(self, max_size=32):
        self.max_size = max_size

    def __call__(self, img):
        w, h = img.size
        scale = self.max_size / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return F_t.resize(img, (new_h, new_w), interpolation=InterpolationMode.BILINEAR)


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
            raise FileNotFoundError(f"[ERROR] Image folder not found: '{image_folder}'")

        if not os.path.isfile(annotation_file):
            raise FileNotFoundError(
                f"[ERROR] Annotation file not found: '{annotation_file}'"
            )

        self.root = image_folder
        self.transforms = transforms

        with open(annotation_file, "r") as f:
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
                + "\n".join(missing_images[:10])
                + ("\n... (more missing)" if len(missing_images) > 10 else "")
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
        orig_width, orig_height = img.size
        anns = self.ann_map.get(img_id, [])

        boxes = []
        labels = []
        masks = []

        for ann in anns:
            x, y, w_box, h_box = ann["bbox"]
            boxes.append([x, y, x + w_box, y + h_box])
            labels.append(ann["category_id"])  # = 1 for traffic sign

            seg = ann.get("segmentation", None)
            if seg is not None:
                if isinstance(seg, list):
                    rles = mask_utils.frPyObjects(seg, orig_height, orig_width)
                    rle = mask_utils.merge(rles)
                else:
                    rle = seg
                mask = mask_utils.decode(rle)
                masks.append(mask)
            else:
                continue

        # ensure that images with zero objects still produce tensors w/ correct shape
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        if labels:
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            labels = torch.zeros((0,), dtype=torch.int64)

        if masks:
            masks = torch.as_tensor(
                np.stack(masks, axis=0), dtype=torch.uint8
            )  # [num_objs, H, W]
        else:
            masks = torch.zeros((0, orig_height, orig_width), dtype=torch.uint8)

        # resize
        if self.transforms:
            img = self.transforms(img)

            new_height, new_width = img.shape[1], img.shape[2]
            scale_x = new_width / orig_width
            scale_y = new_height / orig_height

            if boxes.shape[0] > 0:
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y

            if masks.shape[0] > 0:
                masks_resized = []

                for mask in masks:
                    mask_pil = Image.fromarray(mask.numpy())
                    mask_resized = F_t.resize(
                        mask_pil,
                        (new_height, new_width),
                        interpolation=InterpolationMode.NEAREST,
                    )
                    masks_resized.append(
                        torch.as_tensor(np.array(mask_resized), dtype=torch.uint8)
                    )

                masks = torch.stack(masks_resized)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id]),
        }

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def parse_DFG():
    t_transforms = transforms.Compose(
        [ResizeTransform(max_size=32), transforms.ToTensor()]
    )

    testset = COCOTrafficSigns(
        image_folder="../data/JPEGImages",
        annotation_file="../data/DFG-tsd-annot-json/test.json",
        transforms=t_transforms,
    )
    trainset = COCOTrafficSigns(
        image_folder="../data/JPEGImages",
        annotation_file="../data/DFG-tsd-annot-json/train.json",
        transforms=t_transforms,
    )

    # img, target = trainset[0]
    # print(img.shape) #torch.Size([3, 1080, 1920])

    testloader = DataLoader(
        testset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=4
    )
    trainloader = DataLoader(
        trainset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4
    )

    return trainloader, testloader


def count_classes():
    """
    Returns the number of object classes + 1
    For this dataset, should be 200 + 1
    """

    with open("../data/DFG-tsd-annot-json/test.json") as f:
        data = json.load(f)

    num_obj_classes = len(data["categories"])

    return num_obj_classes + 1


if __name__ == "__main__":
    ### for debugging purposes
    trainloader, testloader = parse_DFG()
    images, targets = next(iter(trainloader))
    print("batch size", len(images))
    print("first image shape:", images[0].shape)
    print("first target:", targets[0])

"""
References: https://towardsdatascience.com/how-to-work-with-object-detection-datasets-in-coco-format-9bf4fb5848a4/
"""
