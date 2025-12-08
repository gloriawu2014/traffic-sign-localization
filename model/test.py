"""
Script for testing our neural network trained in train.py.
"""

import torch
import torchvision
from torchvision.ops import box_iou
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import argparse
import time
from parse_coco import parse_DFG, count_classes

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


def accuracy(model, dataloader, iou: float) -> float:
    """
    Returns the percentage of images with correct bounding boxes in the dataloader.
    "Correct" is defined by having an IoU above the specified amoung, which is passed as an input parameter.
    """
    aps = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            print("Image size:", images[0].shape)
            print("Ground truth boxes:", targets[0]["boxes"])
            print("Predicted boxes:", outputs[0]["boxes"])
            print("Scores:", outputs[0]["scores"])

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

    return sum(aps) / len(aps)


### TODO: SHOW IMAGE WITH BOUNDING BOX


def test(model, testloader, iou: float):
    acc = accuracy(model, testloader, iou)
    print(f"Accuracy (IoU > {iou}) = {acc:.4f}")


if __name__ == "__main__":
    start = time.time()

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

    num_classes = count_classes()
    model = create_mask_rcnn(num_classes)
    state_dict = torch.load(
        "../data/mask_rcnn_traffic_sign_size512_epoch10_weights.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    test(model, testloader, args.iou)

    end = time.time()
    elapsed = end - start
    print(f"Time taken: {elapsed} seconds")
