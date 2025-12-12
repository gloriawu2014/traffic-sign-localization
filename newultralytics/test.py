from ultralytics import YOLO

# Load a YOLO model
model = YOLO(
    "/home/chhlee28/traffic/traffic-sign-localization/ultralytics/runs/detect/train2/weights/best.pt"
)

# Validate on separate data
results = model.val(
    data="/home/chhlee28/traffic/traffic-sign-localization/newultralytics/dfg.yaml",
    project="/home/chhlee28/traffic/traffic-sign-localization/newultralytics/results2",
    name="yolo_test", split="test")

print(results.metrics)
