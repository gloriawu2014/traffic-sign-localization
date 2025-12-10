import os
import json
import shutil
import random


coco_json_path = "/home/chhlee28/traffic/traffic-sign-localization/data/DFG-tsd-annot-json/test.json"
images_src_dir = "/home/chhlee28/traffic/traffic-sign-localization/data/JPEGImages"  # >
output_dir = "dataset"  # Output folder for YOLO dataset
train_ratio = 0.8  # Train/val split ratio


with open(coco_json_path, "r") as f:
    coco_data = json.load(f)


images_info = {img["id"]: img for img in coco_data["images"]}
annotations = coco_data["annotations"]
categories = coco_data["categories"]


category_id_to_yolo_id = {cat["id"]: i for i, cat in enumerate(categories)}
yolo_class_names = [cat["name"] for cat in categories]


# -----------------------------
# CREATE OUTPUT FOLDERS
# -----------------------------
for split in ["test", "val"]:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)


# -----------------------------
# GROUP ANNOTATIONS BY IMAGE
# -----------------------------
annotations_by_image = {}
for ann in annotations:
    img_id = ann["image_id"]
    annotations_by_image.setdefault(img_id, []).append(ann)


# -----------------------------
# SPLIT DATASET
# -----------------------------
image_ids = list(images_info.keys())
random.shuffle(image_ids)
train_cutoff = int(len(image_ids) * train_ratio)
train_ids = set(image_ids[:train_cutoff])
val_ids = set(image_ids[train_cutoff:])


# -----------------------------
# CONVERT TO YOLO FORMAT
# -----------------------------
for img_id in image_ids:
    img_info = images_info[img_id]
    file_name = img_info["file_name"]
    img_w, img_h = img_info["width"], img_info["height"]

    anns = annotations_by_image.get(img_id, [])

    yolo_lines = []

    for ann in anns:
        # Skip crowd or very small boxes
        if ann.get("iscrowd", 0) == 1:
            continue

        # COCO bbox: [x, y, width, height]
        x, y, w, h = ann["bbox"]
        if w < 15 or h < 15:
            continue

        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        class_id = category_id_to_yolo_id[ann["category_id"]]
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    # Determine split
    split = "test" if img_id in train_ids else "val"

    # Copy image
    shutil.copy(os.path.join(images_src_dir, file_name), os.path.join(output_dir, "images", split, file_name))

    # Save label file
    label_file = os.path.join(output_dir, "labels", split, os.path.splitext(file_name)[0] + ".txt")
    with open(label_file, "w") as f:
        f.write("\n".join(yolo_lines))


# -----------------------------
# SAVE CLASS NAMES FILE
# -----------------------------
with open(os.path.join(output_dir, "classes.txt"), "w") as f:
    f.write("\n".join(yolo_class_names))


print("Conversion complete! YOLO dataset ready at:", output_dir)
