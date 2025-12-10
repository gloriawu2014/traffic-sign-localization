"""
Perturb and evaluate model performance on weather perturbations.
All weather perturbations generated using the imagecorruptions package: https://github.com/bethgelab/imagecorruptions
"""

import torch
import torchvision
from torchvision.ops import box_iou
import torchvision.transforms as transforms
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import argparse
import time
import numpy as np

if not hasattr(np, "float_"):
    np.float_ = (
        np.float64
    )  # need NumPy < 2.0, but don't want to change package/venv files
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="pkg_resources is deprecated as an API.*"
)
from imagecorruptions import corrupt


import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.parse_coco import parse_DFG, count_classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def evaluate_corrupt(model, testloader, iou, corruption, severity, num_test):
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

            for output, target in zip(outputs, targets):
                pred_boxes = output["boxes"].cpu()
                true_boxes = target["boxes"]

                if len(pred_boxes) == 0 or len(true_boxes) == 0:
                    continue

                # compute IoU
                ious = box_iou(pred_boxes, true_boxes)
                max_iou_per_gt, _ = ious.max(0)
                num_clean += (max_iou_per_gt > iou).sum().item()

                corrupted_images = []
                for img in images:
                    # corrupt() expects NumPy array of uint8
                    img_np = img.permute(1, 2, 0).cpu().numpy()
                img_np = img_np * std + mean
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                corrupted_np = corrupt(
                    img_np, corruption_name=corruption, severity=severity
                )
                corrupted_tensor = (
                    torch.tensor(corrupted_np / 255.0, dtype=torch.float32)
                    .permute(2, 0, 1)
                    .to(device)
                )
                corrupted_tensor = transforms.Normalize(mean, std)(corrupted_tensor).to(
                    device
                )
                corrupted_images.append(corrupted_tensor)

                corrupted_outputs = model(corrupted_images)

                for corrupted_output, target in zip(corrupted_outputs, targets):
                    corrupt_boxes = corrupted_output["boxes"].cpu()
                    true_boxes = target["boxes"]

                if len(corrupt_boxes) == 0 or len(true_boxes) == 0:
                    continue

                # compute IoU
                ious = box_iou(corrupt_boxes, true_boxes)
                max_iou_per_gt_corrupt, _ = ious.max(0)
                num_correct += (max_iou_per_gt_corrupt > iou).sum().item()

        accuracy = num_correct / num_clean if num_clean > 0 else 0.0
        """print(
            f"Number of correct predictions with IoU {iou}: {num_correct} / {num_clean} = {accuracy:.4f}"
        )"""
        return num_correct, num_clean, accuracy


if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser(
        description="Test model against adversarial weather perturbations"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU tolerance for correct bounding boxes",
    )
    parser.add_argument(
        "--corruption",
        type=str,
        default="snow",
        help="Type of weather perturbation: snow, frost, fog",
    )
    parser.add_argument(
        "--severity", 
        type=int, 
        default=1, 
        help="Severity of corruption"
    )
    parser.add_argument(
        "--num_test", 
        type=int, 
        default=100, 
        help="Number of images to corrupt"
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

    """print(
        f"Evaluating model performance against {args.corruption} perturbations with severity {args.severity}"
    )"""

    correct, total, accuracy = evaluate_corrupt(
        model, testloader, args.iou, args.corruption, args.severity, args.num_test
    )

    end = time.time()
    elapsed = end - start
    # print(f"Time taken: {elapsed} seconds")
    print(f"{args.corruption},{args.severity},{correct},{total},{accuracy},{elapsed}")
