results = model.train(
    data="/content/data/css-data/dataset.yaml",  # Path to the updated dataset.yaml
    epochs=10,                                  # Number of epochs
    imgsz=640,                                  # Image size
    batch=16,                                   # Batch size
    device="0",                                 # Use GPU
    name="yolov8_construction_safety",         # Name of the training run
    workers=4,                                  # Number of data loading workers
    lr0=0.01,                                   # Initial learning rate
    optimizer="auto",                           # Optimizer (auto, SGD, Adam, etc.)
    seed=42                                     # Random seed for reproducibility
)


metrics = model.val()  
print(f"mAP50-95: {metrics.box.map}")  


test_results = model.predict(
    source="/content/data/css-data/test/images", 
    save=True,                                
    conf=0.5,                                    
    iou=0.45,                                 
    show_labels=True,                            
    show_conf=True                               
)

model.save("/content/drive/MyDrive/ConstructionModel/yolov8_construction_safety.pt")
model.export(format="onnx")


test_results[0].show()  
from ultralytics.utils.plots import plot_results
plot_results("/content/runs/detect/yolov8_construction_safety/results.csv")  
