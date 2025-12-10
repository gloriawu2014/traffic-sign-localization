import os
import shutil
import random

# Paths
images_dir = "/home/common/neural-safety-net/traffic-sign/traffic-sign-localization/data/JPEGImages"
labels_dir = "/home/common/neural-safety-net/traffic-sign/traffic-sign-localization/data/DFG-tsd-annot-json"
output_dir = os.path.expanduser("~/traffic_data")

# Split ratio for validation
val_ratio = 0.1

# Create output folders
for split in ["train", "val"]:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

# List all images
images = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
random.shuffle(images)

val_count = int(len(images) * val_ratio)

for i, img_file in enumerate(images):
    split = "val" if i < val_count else "train"

    # Copy image
    shutil.copy(
        os.path.join(images_dir, img_file),
        os.path.join(output_dir, "images", split, img_file),
    )

    # Copy corresponding label
    label_file = os.path.splitext(img_file)[0] + ".txt"
    src_label_path = os.path.join(labels_dir, label_file)
    if os.path.exists(src_label_path):
        shutil.copy(
            src_label_path,
            os.path.join(output_dir, "labels", split, label_file),
        )
    else:
        print(f"Warning: label missing for {img_file}")

print("Dataset split complete!")
