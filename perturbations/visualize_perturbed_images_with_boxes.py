"""
Script to show images with the weather or lighting perturbation, as well as the ground truth and predicted boxes.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import numpy as np

if not hasattr(np, "float_"):
    np.float_ = (
        np.float64
    )  # need NumPy < 2.0, but don't want to change package/venv files
import argparse
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

def getTransformer(lighting_type, severity):
    if lighting_type == "dark":
        transform = transforms.ColorJitter(
            brightness=(0.1 * severity, 0.1 * severity),
            contrast=(0.1 * severity, 0.1 * severity),
            saturation=(0.7 * severity, 0.7 * severity),
        )
    elif lighting_type == "dawn":
        transform = transforms.ColorJitter(
            contrast=(1.1 * severity, 1.1 * severity),
            saturation=(1.2 * severity, 1.2 * severity),
            hue=(0.9 * severity, 0.9 * severity),
        )
    elif lighting_type == "dusk":
        transform = transforms.ColorJitter(
            brightness=(0.7 * severity, 0.7 * severity),
            contrast=(0.8 * severity, 0.8 * severity),
            saturation=(-0.1 * severity, -0.1 * severity),
            hue=(0.1 * severity, 0.1 * severity),
        )
    else:  # "bright" or other
        transform = transforms.ColorJitter(
            brightness=(2 * severity, 2 * severity),
            contrast=(0.5 * severity, 0.5 * severity),
            saturation=(0.8 * severity, 0.9 * severity),
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


def visualize_corruption(
    model,
    dataloader,
    corruption: str,
    severity: int,
    iou: float,
    num_images: int,
    output_dir: str,
    weather: bool
):
    os.makedirs(output_dir, exist_ok=True)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    lighting_transform = none
    if not weather:
        lighting_transform = getTransformer(corruption, severity)

    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            if i >= num_images:
                break

            img = images[0].cpu()
            target = targets[0]

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

            outputs = model([corrupted_tensor])[0]

            img_plot = corrupted_tensor.cpu().permute(1, 2, 0).numpy()
            img_plot = (img_plot - img_plot.min()) / (img_plot.max() - img_plot.min())

            fig, ax = plt.subplots(1, figsize=(10, 10))
            ax.imshow(img_plot)

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
            pred_boxes = outputs["boxes"].cpu()
            scores = outputs["scores"].cpu()
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

            ax.set_title(
                f"Corruption: {corruption}, Severity: {severity} | Yellow: ground truth | Red: predicted"
            )
            ax.axis("off")
            save_path = os.path.join(
                output_dir, f"image_{corruption}_{severity}_{i + 1}.jpg"
            )
            plt.savefig(save_path, format="jpg", dpi=300)
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display images with weather perturbations and the bounding boxes"
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
        help="Type of weather perturbation: snow, frost, fog or lighting: dark, dawn, dusk, bright",
    )
    parser.add_argument(
        "--severity", 
        type=float, 
        default=1, 
        help="Severity of corruption"
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

    if (args.corruption == "snow" or args.corruption == "frost" or args.corruption == "fog"):
        weather = True
    else:
        weather = False
    if weather and not isinstance(args.severity, int):
        print(f"Please provide an integer argument between 1 and 5 for severity")
        exit
    elif not weather and args.severity > 1:
        print(f"Please provide a severity between 0 and 1")
        exit

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

    visualize_corruption(
        model,
        testloader,
        args.corruption,
        args.severity,
        args.iou,
        args.num_images,
        args.output_dir,
        weather
    )
