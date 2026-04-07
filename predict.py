
#image_url = "https://as1.ftcdn.net/v2/jpg/03/31/59/82/1000_F_331598222_tadTdFeI0WxEmyGFoWuMTlEK1jNu2Xhl.jpg"
#!wget -O /content/downloaded_image.jpg {image_url}

from ultralytics import YOLO

# Load the trained model
model = YOLO("/content/drive/MyDrive/ConstructionModel/yolov8_construction_safety.pt")


results = model.predict(
    source="/content/downloaded_image.jpeg",  # Path to the downloaded image
    save=True,                              # Save the predictions
    conf=0.5,                               # Confidence threshold
    show_labels=True,                       # Show labels on predictions
    show_conf=True                          # Show confidence scores
)

results[0].show()
