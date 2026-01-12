from ultralytics import YOLO

# Load a model
model = YOLO("yolo11l.pt")  # load a pretrained model

# Train the model
results = model.train(data="./datasets/data.yaml", epochs=100, imgsz=640)
