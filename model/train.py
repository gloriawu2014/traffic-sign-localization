"""
Script for training a neural network to detect the bounding box of a traffic sign.
The dataset is compatible with Mask-RCNN, so that is what we are using.
Dataset: https://www.vicos.si/resources/dfg/
"""

#### TO-DO: TRY TRAINING WITH DETECTRON2, also to compare

import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import argparse
import time
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


def train(model, trainloader, num_epochs: int, lr: float):
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


if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser(
        description="Train a neural network to detect bounding boxes for traffic signs"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for gradient descent"
    )
    args = parser.parse_args()

    num_classes = count_classes()
    model = create_mask_rcnn(num_classes)
    model.to(device)

    trainloader, _ = parse_DFG()

    train(model, trainloader, args.epochs, args.lr)

    torch.save(
        model.state_dict(), "../data/mask_rcnn_traffic_sign_size512_epoch10_weights.pth"
    )

    end = time.time()
    elapsed = end - start
    print(f"Time taken: {elapsed} seconds")

"""
References:
https://medium.com/analytics-vidhya/training-your-own-data-set-using-mask-r-cnn-for-detecting-multiple-classes-3960ada85079
"""
