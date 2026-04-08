from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.pt")  # load an official detection model
# model = YOLO("yolo11n-seg.pt")  # load an official segmentation model
model = YOLO("runs/detect/train_keke_rev2/weights/best.pt")  # load a custom model

# Track with the model
# results = model.track(source="../Videos/1.mp4", show=True)
# results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")

# Predict only
results = model.predict(source="datasets/test/yola_road_tiny_3s.mp4", show=True, save=True)