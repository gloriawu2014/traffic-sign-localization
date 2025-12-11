"""
Script to visualize images with ground truth and predicted bounding boxes
"""

import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from parse_coco import parse_DFG, count_classes
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import argparse
import os

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


def visualize(model, dataloader, num_images: int, iou: float, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            if i >= num_images:
                break

            img = images[0].cpu()
            target = targets[0]

            img_input = [img.to(device)]
            outputs = model(img_input)

            # convert to NumPy to visualize
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

            fig, ax = plt.subplots(1, figsize=(10, 10))
            ax.imshow(img_np)

            # ground truth box
            for box in target["boxes"]:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor="y",
                    facecolor="none",
                )
                ax.add_patch(rect)

            # predicted box
            pred_boxes = outputs[0]["boxes"].cpu()
            scores = outputs[0]["scores"].cpu()
            for box, score in zip(pred_boxes, scores):
                if score < iou:  # only show if iou threshold is reached
                    continue
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)

            ax.set_title(f"Yellow: ground truth | Red: predicted")
            ax.axis("off")
            save_path = os.path.join(output_dir, f"image{i + 1}.jpg")
            plt.savefig(save_path, format="jpg", dpi=300)
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display images with bounding boxes")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU tolerance for correct bounding boxes",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Number of images to show",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="images",
        help="Output directory for images",
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

    visualize(
        model,
        testloader,
        args.num_images,
        args.iou,
        args.output_dir,
    )
