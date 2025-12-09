from ultralytics import YOLO

# Load a model
model = YOLO("dfg.yaml")

# Train the model
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
