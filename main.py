from ultralytics import YOLO

model = YOLO("/content/drive/MyDrive/ML-project/truck/results/yolo_truck_v1/weights/last.pt")

model.train(
    data="/content/drive/MyDrive/ML-project/truck/data.yaml",   # change this to your path
    epochs=100,
    imgsz=640,
    batch=16,
    project="/content/drive/MyDrive/ML-project/truck/results",
    name="yolo_truck_v1",
    resume=True
)