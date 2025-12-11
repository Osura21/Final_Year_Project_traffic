import numpy as np
import cv2

# =============================
# CONFIG
# =============================
IMG_SIZE = 640
CONF_THRESH = 0.25
IOU_THRESH = 0.45

# IMPORTANT: order must match your data.yaml 'names'
CLASS_NAMES = [
    'Bus-stop', 'Compulsory-Roundabout', 'Cross-Roads-Ahead',
    'Double-Bend-to-Left-Ahead', 'Double-Bend-to-Right-Ahead',
    'Falling-Rocks-Ahead', 'Left-Bend-Ahead',
    'Level-crossing-with-barriers-ahead',
    'Level-crossing-without-barriers-ahead',
    'Narrow-Bridge-or-Culvert-Ahead', 'No-entry', 'No-horns',
    'No-left-turn', 'No-parking', 'No-parking-and-standing',
    'No-parking-on-even-numbered-days',
    'No-parking-on-odd-numbered-days', 'No-right-turn', 'No-u-turn',
    'Pedestrian-Crossing', 'Pedestrian-crossing-Ahead', 'Proceed-straight',
    'Right-Bend-Ahead', 'Road-Bump-Ahead', 'Road-Closed-for-All-Vehicles',
    'Roundabout-Ahead', 'Speed-Limit-40-Kmph', 'Speed-Limit-50-Kmph',
    'Speed-Limit-60-Kmph', 'Stop', 'T-Junction-Ahead',
    'Traffic-from-left-merges-ahead', 'Traffic-from-right-merges-ahead',
    'Turn-left', 'Turn-left-ahead', 'Turn-right', 'Turn-right-ahead'
]


# =============================
# PRE/POST PROCESS HELPERS
# =============================
def letterbox(im, new_shape=IMG_SIZE, color=(114, 114, 114),
              auto=False, scaleFill=False, scaleup=True):
    shape = im.shape[:2]  # current shape [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return im, r, (dw, dh)


def preprocess_image(image_bytes):
    """Convert raw image bytes → model input tensor (1, 3, 640, 640)."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    img0 = img.copy()
    img, r, (dw, dh) = letterbox(img)

    # BGR to BGR (if you want RGB change to img[:, :, ::-1])
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)   # (1, 3, H, W)
    return img0, img, r, dw, dh


def nms(boxes, scores, iou_threshold):
    """Very simple NMS."""
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        iou = compute_iou(boxes[i], boxes[rest])
        idxs = rest[iou < iou_threshold]
    return keep


def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def postprocess(output, img0_shape, r, dw, dh,
                conf_thres=CONF_THRESH, iou_thres=IOU_THRESH):
    """
    YOLOv8 ONNX output (typical):
      - (1, 84, N)  or  (1, N, 84)
      where 84 = 4 box coords + num_classes (no separate obj conf).
    We normalize to (num_boxes, 4 + num_classes).
    """
    preds = output  # this is already a numpy array in your code

    # Expect 3D: (1, C, N) or (1, N, C)
    if preds.ndim == 3:
        preds = np.squeeze(preds, axis=0)  # (C, N) or (N, C)

        # If first dim is "channels" (e.g. 42, 84) and second is big (e.g. 8400),
        # transpose to (N, C).
        if preds.shape[0] <= preds.shape[1]:
            # e.g. (42, 8400) -> (8400, 42)
            preds = preds.transpose(1, 0)
    elif preds.ndim != 2:
        raise ValueError(f"Unexpected preds shape: {preds.shape}")

    # Now preds is (num_boxes, 4 + num_classes)
    if preds.shape[1] <= 4:
        raise ValueError(f"Not enough channels in output: {preds.shape}")

    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]  # no separate obj conf for YOLOv8

    # per-box class prediction
    class_ids = np.argmax(class_scores, axis=1)
    final_scores = class_scores[np.arange(len(class_scores)), class_ids]

    # filter by confidence
    mask = final_scores > conf_thres
    boxes_xywh = boxes_xywh[mask]
    final_scores = final_scores[mask]
    class_ids = class_ids[mask]

    if len(final_scores) == 0:
        return []

    # xywh → xyxy
    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    # undo letterbox scale/pad
    h0, w0 = img0_shape[:2]
    boxes_xyxy[:, [0, 2]] -= dw
    boxes_xyxy[:, [1, 3]] -= dh
    boxes_xyxy[:, [0, 2]] /= r
    boxes_xyxy[:, [1, 3]] /= r

    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, w0)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, h0)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, w0)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, h0)

    # NMS
    keep = nms(boxes_xyxy, final_scores, iou_thres)

    detections = []
    for i in keep:
        x1, y1, x2, y2 = boxes_xyxy[i]
        cls_id = int(class_ids[i])
        # safe label lookup
        if 0 <= cls_id < len(CLASS_NAMES):
            cls_name = CLASS_NAMES[cls_id]
        else:
            cls_name = f"class_{cls_id}"

        detections.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "class_name": cls_name,
            "score": float(final_scores[i]),
            "class_id": cls_id
        })
    return detections
