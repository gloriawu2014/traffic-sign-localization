import json
import os
import shutil

# -------------------- PATHS --------------------
JSON_DIR = "/home/chhlee28/traffic/traffic-sign-localization/data/DFG-tsd-annot-json"
OUTPUT_DIR = "/home/chhlee28/traffic/traffic-sign-localization/newultralytics/dataset"

# Path where the images actually live
IMG_SOURCE = "/home/chhlee28/traffic/traffic-sign-localization/data/DFG-tsd-annot-json/JPEGImages"

# -------------------- DATA SPLITS --------------------
SETS = {
    "train": "train.json",
    "test": "test.json"
}

# -------------------- CREATE FOLDERS --------------------
for split in SETS.keys():
    os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

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

        label_path = f"{OUTPUT_DIR}/labels/{split}/{file_name.replace('.jpg', '.txt').replace('.png', '.txt')}"
        with open(label_path, "a") as f:
            f.write(f"{cls} {xc} {yc} {nw} {nh}\n")

print("\nConversion complete!")
print(f"Dataset saved in: {OUTPUT_DIR}")
