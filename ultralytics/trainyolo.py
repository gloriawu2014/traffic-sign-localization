from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML

# Train the model
results = model.train(data="dfg.yaml", epochs=100, imgsz=640)
