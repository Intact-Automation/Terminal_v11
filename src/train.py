from ultralytics import YOLO

model = YOLO('src/config/yolov11-p2.yaml')

model.train(
    data='src/dataset/data.yaml',
    epochs=800,
    batch=256,
    imgsz=1920,
    workers=16,
    close_mosaic=20,
    cos_lr=True,
    device=0,
    hyp='src/config/augment.yaml'   # load this file
)

# 3️⃣ Evaluate model performance on the validation set
metrics = model.val()
