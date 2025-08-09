import os, random
import numpy as np
import torch, cv2
from ultralytics import YOLO

# (optional) silence the write warning when running as root
os.environ['YOLO_CONFIG_DIR'] = '/tmp/ultra_cfg'
os.makedirs('/tmp/ultra_cfg', exist_ok=True)

# ---- custom batch augmentation via callback ----
# Applies: grayscale (p=0.2), Gaussian blur (p=0.3), CLAHE (p=0.3), sharpen (p=0.3)
def on_preprocess_batch(trainer):
    batch = trainer.batch
    imgs = batch["img"]  # (B, C, H, W) float32 in [0,1]
    B, C, H, W = imgs.shape
    for i in range(B):
        # to uint8 BGR for OpenCV
        img = (imgs[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # grayscale 20% (keep 3 channels)
        if random.random() < 0.2:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

        # gaussian blur 30%
        if random.random() < 0.3:
            img = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=1.0)

        # CLAHE 30% (on L channel)
        if random.random() < 0.3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            L, A, Bc = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            L2 = clahe.apply(L)
            lab2 = cv2.merge([L2, A, Bc])
            img = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

        # sharpen 30% (unsharp mask)
        if random.random() < 0.3:
            blur = cv2.GaussianBlur(img, (0, 0), 1.0)
            img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

        # back to tensor in [0,1]
        img_t = torch.from_numpy(img).permute(2, 0, 1).to(imgs.device).float() / 255.0
        imgs[i] = img_t

    batch["img"] = imgs

callbacks = {"on_preprocess_batch": on_preprocess_batch}
# -----------------------------------------------

# load P2-head model yaml
model = YOLO("src/config/yolov11-p2.yaml")

# train
model.train(
    data="src/dataset/data.yaml",
    epochs=800,
    batch=256,        # you have 180 GB VRAM
    imgsz=1920,
    workers=16,
    mosaic=1.0,
    close_mosaic=20,
    hsv_h=0.015, hsv_s=0.6, hsv_v=0.5,
    translate=0.1, scale=0.5, fliplr=0.5, mixup=0.0,
    cos_lr=True,
    device=0,
    callbacks=callbacks,   # << inject our augmentations here
)

# validate
metrics = model.val()

# quick inference test
result = model("path/to/image.jpg")
result[0].show()
