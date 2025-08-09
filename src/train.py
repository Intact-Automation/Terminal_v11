import albumentations as A
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.cfg import get_cfg


# ===== 1️⃣ Custom Dataset with Albumentations =====
class ScrewDataset(YOLODataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Albumentations pipeline
        self.albu = A.Compose([
            A.ToGray(p=0.2),  # 20% grayscale images
            A.GaussianBlur(blur_limit=3, p=0.3),  # mild blur
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),  # contrast boost
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3)  # edge sharpening
        ])

    def __getitem__(self, index):
        im, labels = super().__getitem__(index)
        h, w = im.shape[1], im.shape[2]

        # Convert to numpy for Albumentations (H, W, C)
        im = im.permute(1, 2, 0).numpy()
        im = self.albu(image=im)['image']

        # Convert back to tensor (C, H, W)
        im = self.transforms(im)  # use YOLO's transforms to normalize
        return im, labels


# ===== 2️⃣ Load model =====
model = YOLO('src/config/yolov11-p2.yaml')  # make sure this yaml exists

# ===== 3️⃣ Training config =====
cfg = get_cfg(cfg='src/config/default.yaml')  # load default config
cfg.data = 'src/dataset/data.yaml'
cfg.epochs = 800
cfg.batch = 256
cfg.imgsz = 1920
cfg.workers = 16
cfg.close_mosaic = 20
cfg.cos_lr = True
cfg.device = 0

# ===== 4️⃣ Train with custom dataset =====
model.train(cfg=cfg, dataset=ScrewDataset)

# ===== 5️⃣ Validate =====
metrics = model.val()

# ===== 6️⃣ Inference example =====
results = model("path/to/image.jpg")
results[0].show()
