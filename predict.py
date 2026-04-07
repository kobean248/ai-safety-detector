import argparse
from ultralytics import YOLO

def predict(model_path, source, conf=0.5, iou=0.45):
    model = YOLO(model_path)
    results = model.predict(
        source=source,
        save=True,
        conf=conf,
        iou=iou,
        show_labels=True,
        show_conf=True
    )
    results[0].show()
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8_construction_safety.pt")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    predict(args.model, args.source, args.conf, args.iou)
