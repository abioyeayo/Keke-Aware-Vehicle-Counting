from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11l.pt")
    parser.add_argument("--data", default="./datasets/data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=0)
    # parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="train")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        # project=args.project,
        name=args.name,
    )

if __name__ == "__main__":
    main()


# from ultralytics import YOLO
# # Load a model
# model = YOLO("yolo11l.pt")  # load a pretrained model
# # Train the model
# results = model.train(data="./datasets/data.yaml", epochs=100, imgsz=640)

# python train.py --epochs 100 --imgsz 640 --name train_keke_rev1