from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")
results = model.train(data="dfg.yaml", epochs=50, imgsz=640)
