from ultralytics import YOLO

# Load a YOLO model
model = YOLO("/home/common/neural-safety-net/traffic-sign/traffic-sign-localization/ultralytics/runs/detect/train/weights/best.pt")

# Validate on separate data
model.val(
    data="/home/chhlee28/traffic/traffic-sign-localization/newultralytics/dfgtest.yaml",
    project="/home/chhlee28/traffic/results",
    name="dfg_val",
)
