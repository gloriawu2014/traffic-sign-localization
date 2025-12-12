from ultralytics import YOLO

# Load a YOLO model
model = YOLO(
    "/home/common/neural-safety-net/traffic-sign/traffic-sign-localization/ultralytics/runs/detect/train8/weights/best.pt"
)

# Validate on separate data
results = model.val(
    data="/home/chhlee28/traffic/traffic-sign-localization/newultralytics/dfgtest.yaml", split="test",
    project="/home/chhlee28/traffic/traffic-sign-localization/newultralytics/results2",
    name="yolo_test", save=True)
