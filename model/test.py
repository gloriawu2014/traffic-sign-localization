"""
Script for testing our neural network trained in train.py.
"""

import torch
import torchvision
import argparse
from parse_coco import parse_DFG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(model, dataloader, iou: float) -> float:
    """
    Returns the percentage of images with correct bounding boxes in the dataloader.
    "Correct" is defined by having an IoU above the specified amoung, which is passed as an input parameter.
    """
    is_training = model.training
    model.eval()
    aps = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                pred_boxes = output["boxes"].cpu()
                true_boxes = target["boxes"]

                # edge case: no predicted boxes
                if len(pred_boxes) == 0:
                    aps.append(0.0)
                    continue

                # compute IoU
                ious = box_iou(pred_boxes, true_boxes)
                max_ious = ious.max(dim=0).values
                correct = (max_ious > iou).sum().item()
                total = len(true_boxes)

                aps.append(correct / total)

    if is_training:
        model.train()

    return sum(aps) / len(aps)


### TODO: SHOW IMAGE WITH BOUNDING BOX


def test(model, testloader, iou: float):
    acc = accuracy(model, trainloader, iou)
    print(f"Accuracy (IoU > {iou}) = {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a neural network to detect bounding boxes for traffic signs"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU tolerance for correct bounding boxes",
    )
    args = parser.parse_args()

    _, testloader = parse_DFG()

    model = torch.load("data/mask_rcnn_traffic_sign_epoch_10.pth")

    train(model, testloader, args.iou)
