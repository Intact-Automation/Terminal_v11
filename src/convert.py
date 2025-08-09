from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
model.export(format="engine", half=True, dynamic=True, workspace=4)
