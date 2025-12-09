"""
Perturb and evaluate model performance on weather perturbations.
All weather perturbations generated using the imagecorruptions package: https://github.com/bethgelab/imagecorruptions
"""

import torch
import torchvision
from torchvision.ops import box_iou
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from imagecorruptions import corrupt
from parse_coco import parse_DFG, count_classes
import argparse
import time

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
    with torch.no_grad():
        for image, target in dataloader:
            if num_clean == num_test:
                break

            image = [img.to(device) for img in image]
            output = model(image)

            ### only corrupt clean images
            pred_boxes = output["boxes"].cpu()
            true_boxes = target["boxes"]

            # edge case: no predicted boxes
            if len(pred_boxes) == 0:
                continue

            # edge case: no ground truth boxes
            if len(true_boxes) == 0:
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
                    num_clean += 1
                    continue

            corrupted_image = corrupt(image, corruption_name=corruption, severity=1)
            corrupted_output = model(corrupted_image)

            corrupt_boxes = output["boxes"].cpu()

            # edge case: no predicted boxes
            if len(corrupt_boxes) == 0:
                continue

            # compute IoU
            ious = box_iou(corrupt_boxes, true_boxes)
            matched = torch.zeros(len(true_boxes), dtype=torch.bool)
            for i in range(len(pred_boxes)):
                iou_vals = ious[i]
                iou_vals = iou_vals.clone()
                iou_vals[matched] = -1
                max_iou, max_idx = iou_vals.max(0)
                if max_iou > iou:
                    matched[max_idx] = True
                    num_correct += 1
                    continue

        print(f"Number of correct predictions with IoU {iou}: {num_correct} / {num_clean} = {num_correct/num_clean:.4f}")


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
        description="Type of weather perturbation: snow, frost, fog",
    )
    parser.add_argument(
        "--severity", type=int, default=1, description="Severity of corruption"
    )
    parser.add_argument("--num_test", type=int, default=20, description="Number of images to corrupt")
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

    evaluate_corrupt(model, testloader, args.iou, args.corruption, args.severity, args.num_test)

    end = time.time()
    elapsed = end - start
    print(f"Time taken: {elapsed} seconds")
