from ultralytics import YOLO

# 1️⃣ Load YOLOv11x-P2 model
model = YOLO('src/config/yolov11-p2.yaml')

# 2️⃣ Train
model.train(
    data='src/dataset/data.yaml',
    cfg='src/config/augment.yaml',   # load augmentations from file
    epochs=800,
    batch=256,
    imgsz=1920,
    workers=16,
    close_mosaic=20,
    cos_lr=True,
    device=0
)

# 3️⃣ Evaluate model performance on the validation set
metrics = model.val()
