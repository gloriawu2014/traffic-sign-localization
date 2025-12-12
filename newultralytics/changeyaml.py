import json
import os
import shutil
import random

# -------------------- PATHS --------------------
JSON_DIR = "/home/chhlee28/traffic/traffic-sign-localization/data/DFG-tsd-annot-json"
OUTPUT_DIR = "/home/chhlee28/traffic/traffic-sign-localization/newultralytics/dataset"

# Path where the images actually live
IMG_SOURCE = "/home/chhlee28/traffic/traffic-sign-localization/data/JPEGImages"

# -------------------- DATA SPLITS --------------------
SETS = {
    "train": "train.json",
    "test": "test.json"
}

VAL_PERCENT = 0.1  # 5% of train images will be used for val

# -------------------- CREATE FOLDERS --------------------
for split in SETS.keys():
    os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

# Create val folders
os.makedirs(f"{OUTPUT_DIR}/images/val", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels/val", exist_ok=True)

# -------------------- CLASS MAPPING --------------------
with open(f"{JSON_DIR}/train.json") as f:
    train_data = json.load(f)

# Map category IDs → YOLO class numbers
catid2cls = {cat["id"]: i for i, cat in enumerate(train_data["categories"])}
print("Class mapping:", catid2cls)

# -------------------- CONVERSION --------------------
for split, json_file in SETS.items():
    print(f"\nProcessing {split}...")
    with open(f"{JSON_DIR}/{json_file}") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    anns = data["annotations"]

    for ann in anns:
        img_id = ann["image_id"]
        img_info = images[img_id]

        file_name = img_info["file_name"]

        # -------- Copy Image --------
        src = f"{IMG_SOURCE}/{file_name}"
        dst = f"{OUTPUT_DIR}/images/{split}/{file_name}"
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Warning: {src} not found")

        # -------- Create YOLO Label --------
        width = img_info["width"]
        height = img_info["height"]

        x, y, w, h = ann["bbox"]

        # Convert to YOLO format
        xc = (x + w / 2) / width
        yc = (y + h / 2) / height
        nw = w / width
        nh = h / height

        cls = catid2cls[ann["category_id"]]

        label_path = f"{OUTPUT_DIR}/labels/{split}/{file_name.replace('.jpg', '.txt')}"
        with open(label_path, "a") as f:
            f.write(f"{cls} {xc} {yc} {nw} {nh}\n")

# -------------------- CREATE VAL FROM TRAIN --------------------
train_images = os.listdir(f"{OUTPUT_DIR}/images/train")
num_val = max(1, int(len(train_images) * VAL_PERCENT))
val_samples = random.sample(train_images, num_val)

for img_file in val_samples:
    # Move images
    shutil.move(f"{OUTPUT_DIR}/images/train/{img_file}", f"{OUTPUT_DIR}/images/val/{img_file}")
    # Move labels
    label_file = img_file.replace(".jpg", ".txt")
    shutil.move(f"{OUTPUT_DIR}/labels/train/{label_file}", f"{OUTPUT_DIR}/labels/val/{label_file}")

print("\nConversion complete!")
print(f"Dataset saved in: {OUTPUT_DIR}")
print(f"Created {num_val} validation images in 'val' folder.")
