import argparse
from ultralytics import YOLO

def train(data, epochs, batch, device, imgsz):
    model = YOLO("yolov8n.pt")
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        name="yolov8_construction_safety",
        workers=4,
        lr0=0.01,
        optimizer="auto",
        seed=42
    )

    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")

    results = model.predict(
        source="data/css-data/test/images",
        save=True,
        conf=0.5,
        iou=0.45,
        show_labels=True,
        show_conf=True
    )
    results[0].show()

    model.save("yolov8_construction_safety.pt")
    model.export(format="onnx")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataset.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    train(args.data, args.epochs, args.batch, args.device, args.imgsz)
