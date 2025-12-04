"""
Script for training a neural network to detect the bounding box of a traffic sign.
The dataset is compatible with Mask-RCNN, so that is what we are using.
Dataset: https://www.vicos.si/resources/dfg/
"""

#### TO-DO: TRY TRAINING WITH DETECTRON2, also to compare

import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import box_iou
import argparse
from parse_coco import parse_DFG, count_classes
from PIL import Image

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


def train(model, trainloader, num_epochs: int, lr: float, iou: float):
    """
    Takes in the number of epochs and learning rate as input parameters and trains the model.
    """
    print(f"Training model with {num_epochs} epochs and learning rate {lr} " + "-" * 20)

    optimizer = torch.optim.Adam(model.parameters(), lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, targets in trainloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            losses = model(images, targets)
            # loss = sum(losses.values())
            loss = sum(loss for loss in losses.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}\t Loss = {total_loss:.4f}")

        # acc = accuracy(model, trainloader, iou)
        # print(f"Accuracy (IoU > {iou}) = {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a neural network to detect bounding boxes for traffic signs"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for gradient descent"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU tolerance for correct bounding boxes",
    )
    args = parser.parse_args()

    num_classes = count_classes()
    model = create_mask_rcnn(num_classes)
    model.to(device)

    trainloader, testloader = parse_DFG()

    train(model, trainloader, args.epochs, args.lr, args.iou)

    torch.save(model, "data/mask_rcnn_traffic_sign.pth")

"""
References:
https://medium.com/analytics-vidhya/training-your-own-data-set-using-mask-r-cnn-for-detecting-multiple-classes-3960ada85079
"""
