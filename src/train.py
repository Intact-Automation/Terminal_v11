from ultralytics import YOLO

# 1️⃣ Load YOLOv11x-P2 model
model = YOLO('yolov11-p2.yaml')

# 2️⃣ Train
model.train(
    data='src/dataset/data.yaml',   # Dataset YAML
    epochs=800,                     # Long training for subtle differences
    batch=256,                      # Large batch size for stability
    imgsz=1920,                      # High res for small screws
    workers=16,                      # Speed up dataloading
    mosaic=1.0,                      # Keep mosaic early
    close_mosaic=20,                 # Turn off mosaic in last 20 epochs
    hsv_s=0.6,                       # Saturation aug
    hsv_v=0.5,                       # Brightness aug
    grayscale=0.2,                   # 20% grayscale images
    blur=1.5,                        # Mild blur to simulate focus issues
    cos_lr=True,                     # Smooth learning rate decay
    device=0,                        # GPU index
)

# 3️⃣ Evaluate model performance on the validation set
metrics = model.val()
