import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import time
from torchvision.ops import box_iou
import argparse
import numpy as np
from PIL import Image

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.parse_coco import parse_DFG, count_classes

"""
Perturb and evaluate model performance on lighting perturbations.
All lighting perturbations generated using the ColorJitter module from TorchVision: https://docs.pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# creates a transformer to make an image look more bright or dark or more like dawn or dusk
# severity should range from 0 to 1.0, and is how severely bright/dusky/etc.
def getTransformer(lighting_type, severity):
    if lighting_type == "dawn":
        transform = transforms.ColorJitter(
            contrast=(1.1 * severity, 1.1 * severity),
            saturation=(1.2 * severity, 1.2 * severity),
            hue=(-0.1 * severity, -0.1 * severity),
        )
    elif lighting_type == "dusk":
        transform = transforms.ColorJitter(
            brightness=(0.7 * severity, 0.7 * severity),
            contrast=(0.8, 0.8),
            saturation=(0.5 * severity, 0.5 * severity),
            hue=(0.1 * severity, 0.1 * severity),
        )
    else:  # "bright" or other
        transform = transforms.ColorJitter(
            brightness=(2 * severity, 2 * severity),
        )
    return transform


def create_mask_rcnn(num_classes: int):
    """
    Loads and modifies a pre-trained Mask R-CNN model with a specified
    number of classes and then returns the modified model.
    """
    # load pretrained model - too large for local, train on HPC
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
    )

    # input channels to mask predictor -> intermediate layer in mask head -> new mask prediction head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256  # intermediate layer
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def evaluate_corrupt(model, testloader, iou, transform, num_test):
    num_clean = 0
    num_correct = 0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    with torch.no_grad():
        for images, targets in testloader:
            if num_clean >= num_test:
                break

            images = [img.to(device) for img in images]
            outputs = model(images)

            for img, output, target in zip(images, outputs, targets):
                pred_boxes = output["boxes"].cpu()
                true_boxes = target["boxes"]

                if len(pred_boxes) == 0 or len(true_boxes) == 0:
                    continue

                # compute IoU
                ious = box_iou(pred_boxes, true_boxes)
                max_iou_per_gt, _ = ious.max(0)
                num_clean += (max_iou_per_gt > iou).sum().item()

                img_np = img.permute(1, 2, 0).cpu().numpy()
                img_np = img_np * std + mean
                img_np = np.clip(img_np, 0, 1)

                pil = transforms.ToPILImage()(img_np)
                jittered = transform(pil)
                corrupted_tensor = transforms.ToTensor()(jittered)
                corrupted_tensor = transforms.Normalize(mean, std)(corrupted_tensor).to(
                    device
                )

                corrupted_output = model([corrupted_tensor])[0]
                corrupt_boxes = corrupted_output["boxes"].cpu()

                if len(corrupt_boxes) == 0 or len(true_boxes) == 0:
                    continue

                # compute IoU
                ious_corrupt = box_iou(corrupt_boxes, true_boxes)
                max_iou_per_gt_corrupt, _ = ious_corrupt.max(0)
                num_correct += (max_iou_per_gt_corrupt > iou).sum().item()

        accuracy = num_correct / num_clean if num_clean > 0 else 0.0
        """print(
            f"Number of correct predictions with IoU {iou}: {num_correct} / {num_clean} = {accuracy:.4f}"
        )"""
        return num_correct, num_clean, accuracy


if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser(
        description="Test model against adversarial lighting perturbations"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU tolerance for correct bounding boxes",
    )
    parser.add_argument(
        "--lighting_type",
        type=str,
        default="bright",
        help="Type of lighting perturbation: dawn, dusk, bright, dark",
    )
    parser.add_argument(
        "--severity",
        type=float,
        default=0.5,
        help="Severity of corruption from 0.0 to 1.0",
    )
    parser.add_argument(
        "--num_test", type=int, default=20, help="Number of images to corrupt"
    )
    args = parser.parse_args()

    _, testloader = parse_DFG()

    num_classes = count_classes()
    model = create_mask_rcnn(num_classes)
    state_dict = torch.load(
        "../data/final_model.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    correct, total, accuracy = evaluate_corrupt(
        model,
        testloader,
        args.iou,
        getTransformer(args.lighting_type, args.severity),
        args.num_test,
    )

    end = time.time()
    elapsed = end - start
    # print(f"Time taken: {elapsed} seconds")
    print(
        f"{args.lighting_type},{args.severity},{correct},{total},{accuracy},{elapsed}"
    )
