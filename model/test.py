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
            # print("dtype:", images[0].dtype)
            # print("min/max:", images[0].min().item(), images[0].max().item())

            images = [img.to(device) for img in images]
            outputs = model(images)

            # print("Image size:", images[0].shape)
            # print("Ground truth boxes:", targets[0]["boxes"])
            # print("Predicted boxes:", outputs[0]["boxes"])

            for output, target in zip(outputs, targets):
                pred_boxes = output["boxes"].cpu()
                true_boxes = target["boxes"]

                # edge case: no predicted boxes
                if len(pred_boxes) == 0:
                    #print("no pred")
                    aps.append(0.0)
                    continue

                # edge case: no ground truth boxes
                if len(true_boxes) == 0:
                    #print("no gt")
                    aps.append(1.0)
                    continue

                # compute IoU
                ious = box_iou(pred_boxes, true_boxes)
                matched = torch.zeros(len(true_boxes), dtype=torch.bool)
                for i in range(len(pred_boxes)):
                    iou_vals = ious[i]
                    iou_vals = iou_vals.clone()
                    iou_vals[matched] = -1
                    max_iou, max_idx = iou_vals.max(0)
                    if max_iou > iou:
                        matched[max_idx] = True

                # max_ious = ious.max(dim=0).values
                # correct = (max_ious > iou).sum().item()
                # total = len(true_boxes)

                # aps.append(correct / total)
                aps.append(matched.sum().item() / len(true_boxes))

    return sum(aps) / len(aps)


### TODO: SHOW IMAGE WITH BOUNDING BOX


def test(model, testloader, iou: float):
    acc = accuracy(model, testloader, iou)
    print(f"Accuracy (IoU > {iou}) = {100*acc:.4f}%")


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
        "../data/final_model.pth",
        map_location=device,
    )
    # print("Loaded weights keys:", len(state_dict.keys()))
    # missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # print("Missing keys:", missing)
    # print("Unexpected keys:", unexpected)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # dummy = torch.rand(3, 512, 512).to(device)
    # pred = model([dummy])[0]
    # print("Dummy boxes:", pred['boxes'])
    # print("Dummy scores:", pred['scores'])

    test(model, testloader, args.iou)

    end = time.time()
    elapsed = end - start
    print(f"Time taken: {elapsed} seconds")
