from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

# Use a fixed size (match your inference size; try 640 first)
model.export(
    format="engine",
    half=True,
    dynamic=False,  # <- important
    imgsz=640,  # try 512 if you still hit issues / want more FPS
    batch=1,
    workspace=4,
)
# Expect: runs/detect/train/weights/best.engine
