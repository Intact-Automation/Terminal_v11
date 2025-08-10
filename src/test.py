from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
import torch

# ========= CONFIG =========
ENGINE_PATH = "runs/detect/train/weights/best.engine"  # Model weights
CAM_INDEX = 0  # your camera index
IMG_TILE = 640  # per-tile inference size
OVERLAP = 0.25  # 25% overlap between tiles
CENTER_MARGIN = 0.10  # keep boxes whose centers are 10% away from tile edges
IOU_MERGE = 0.45  # IoU to merge tile detections (reduced for better merging)
AREA_MIN = 0.01  # min box area as fraction of full frame
AREA_MAX = 0.25  # max box area as fraction of full frame
REJECT_MIN_VOTES = 2  # #tiles that must agree on "Reject"

# Auto-detect device (CUDA if available, otherwise CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Debug mode for troubleshooting
DEBUG_MODE = True  # Set to True to see debug information

# class IDs in your dataset:
CLS_INVERTED, CLS_PASS, CLS_REJECT = 0, 1, 2
# class-specific thresholds (stricter Reject)
CLS_THRESH = {CLS_INVERTED: 0.40, CLS_PASS: 0.30, CLS_REJECT: 0.60}
NAMES = {0: "Inverted", 1: "Pass", 2: "Reject"}


# ========= HELPERS =========
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xx1, yy1 = max(ax1, bx1), max(ay1, by1)
    xx2, yy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def keep_center_in_tile(box_xyxy, tw, th, margin=CENTER_MARGIN):
    x1, y1, x2, y2 = box_xyxy
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    return (margin * tw <= cx <= (1 - margin) * tw) and (
        margin * th <= cy <= (1 - margin) * th
    )


def tile_coords(w, h, tile=IMG_TILE, overlap=OVERLAP):
    step = int(tile * (1 - overlap))
    xs = list(range(0, max(1, w - tile + 1), step)) or [0]
    ys = list(range(0, max(1, h - tile + 1), step)) or [0]
    if xs[-1] + tile < w:
        xs.append(max(0, w - tile))
    if ys[-1] + tile < h:
        ys.append(max(0, h - tile))
    for y in ys:
        for x in xs:
            yield x, y, min(tile, w - x), min(tile, h - y)


def merge_detections(dets, iou_thr=IOU_MERGE, reject_min_votes=REJECT_MIN_VOTES):
    """dets: list of (xyxy_global, cls, conf)"""
    if not dets:
        return []

    used = [False] * len(dets)
    merged = []

    for i, a in enumerate(dets):
        if used[i]:
            continue

        # Find all detections that overlap with current detection
        group_idx = [i]
        for j in range(i + 1, len(dets)):
            if used[j]:
                continue
            if iou_xyxy(a[0], dets[j][0]) >= iou_thr:
                group_idx.append(j)

        # Extract data for the group
        boxes = np.array([dets[k][0] for k in group_idx], dtype=float)
        clss = [int(dets[k][1]) for k in group_idx]
        confs = np.array([dets[k][2] for k in group_idx], dtype=float)

        # Weighted average of box coordinates based on confidence
        weights = confs / (confs.sum() + 1e-6)
        xyxy = (boxes * weights[:, None]).sum(axis=0)

        # Vote for class based on frequency
        vote = Counter(clss)
        voted_cls, votes = vote.most_common(1)[0]

        # Special handling for REJECT class - require minimum votes
        if voted_cls == CLS_REJECT and votes < reject_min_votes:
            clss_wo_rej = [c for c in clss if c != CLS_REJECT]
            if clss_wo_rej:
                voted_cls = Counter(clss_wo_rej).most_common(1)[0][0]

        # Use mean confidence for the group
        conf = float(confs.mean())

        # Apply class-specific threshold
        thr = CLS_THRESH.get(voted_cls, 0.5)
        if conf >= thr:
            merged.append((xyxy, voted_cls, conf))

        # Mark all detections in this group as used
        for k in group_idx:
            used[k] = True

    return merged


def dedupe_conflicts(dets, iou_thr=0.6, prefer="pass_if_close", delta=0.07):
    """
    Ensure one final box per object. If two classes overlap, resolve:
    - prefer='higher': keep higher confidence only
    - prefer='pass_if_close': if |conf diff| <= delta, pick PASS over REJECT
    """
    if not dets:
        return []

    kept = []
    dets = sorted(dets, key=lambda d: d[2], reverse=True)

    for di in dets:
        xi, ci, pi = di
        should_keep = True
        replace_idx = None

        for idx, (xk, ck, pk) in enumerate(kept):
            iou = iou_xyxy(xi, xk)
            if iou >= iou_thr:
                # Same class - keep higher confidence (already sorted)
                if ci == ck:
                    should_keep = False
                    break

                # Different classes - apply conflict resolution
                if prefer == "pass_if_close" and abs(pi - pk) <= delta:
                    # If confidences are close, prefer PASS over REJECT
                    if ci == CLS_PASS and ck == CLS_REJECT:
                        replace_idx = idx
                        break
                    elif ck == CLS_PASS and ci == CLS_REJECT:
                        should_keep = False
                        break

                # For other conflicts or when confidences differ significantly,
                # keep the higher confidence one (first in sorted list)
                should_keep = False
                break

        if should_keep:
            if replace_idx is not None:
                # Replace the existing detection with the new PASS detection
                kept[replace_idx] = di
            else:
                kept.append(di)

    return kept


def draw_detections(img, dets, names=None):
    vis = img.copy()
    for (x1, y1, x2, y2), c, p in dets:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        color = (
            (0, 255, 0)
            if c == CLS_PASS
            else ((0, 165, 255) if c == CLS_INVERTED else (0, 0, 255))
        )
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{names[c] if names else c}:{p:.2f}"
        cv2.putText(
            vis, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
    return vis


# ========= MODEL & LOOP =========
try:
    model = YOLO(ENGINE_PATH)  # load PyTorch model weights
    print(f"Successfully loaded model from {ENGINE_PATH}")
except Exception as e:
    print(f"Error loading model from {ENGINE_PATH}: {e}")
    print("Please check if the model file exists and is valid.")
    raise SystemExit

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    print("Trying to use built-in camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No camera found. Please check your camera connection.")
        raise SystemExit

while True:
    ok, frame = cap.read()
    if not ok:
        break

    H, W = frame.shape[:2]
    full_area = float(W * H)
    dets_global = []

    # tile over frame
    for x, y, tw, th in tile_coords(W, H, tile=IMG_TILE, overlap=OVERLAP):
        tile = frame[y : y + th, x : x + tw]

        r = model.predict(
            tile,
            imgsz=IMG_TILE,
            conf=0.20,
            iou=0.60,
            agnostic_nms=True,
            device=DEVICE,
            verbose=False,
        )[0]
        b = r.boxes
        if b is None or b.xyxy is None:
            continue

        xyxy = b.xyxy.cpu().numpy()
        conf = b.conf.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)

        keep_idx = []
        for i in range(len(xyxy)):
            if not keep_center_in_tile(xyxy[i], tw, th):
                continue
            x1, y1, x2, y2 = xyxy[i]
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            frac = area / full_area
            if frac < AREA_MIN or frac > AREA_MAX:
                continue
            keep_idx.append(i)

        for i in keep_idx:
            gx1, gy1, gx2, gy2 = xyxy[i] + np.array([x, y, x, y], dtype=float)
            c = int(cls[i])
            p = float(conf[i])
            # lenient per-tile check; final thresholding happens after merge
            if p >= max(0.20, CLS_THRESH.get(c, 0.5) - 0.20):
                dets_global.append((np.array([gx1, gy1, gx2, gy2], dtype=float), c, p))

    # merge overlapping tile detections, then dedupe cross-class conflicts
    if DEBUG_MODE:
        print(f"Raw detections from tiles: {len(dets_global)}")

    merged = merge_detections(
        dets_global, iou_thr=IOU_MERGE, reject_min_votes=REJECT_MIN_VOTES
    )

    if DEBUG_MODE:
        print(f"After merging: {len(merged)}")

    merged = dedupe_conflicts(merged, iou_thr=0.45, prefer="pass_if_close", delta=0.07)

    if DEBUG_MODE:
        print(f"After deduplication: {len(merged)}")
        for i, (box, cls, conf) in enumerate(merged):
            print(
                f"  Box {i}: {NAMES[cls]} conf={conf:.3f} bbox=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]"
            )

    vis = draw_detections(frame, merged, NAMES)
    cv2.imshow("YOLOv11-P2 Tiled Inference (Orin)", vis)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
