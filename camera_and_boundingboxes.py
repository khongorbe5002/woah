import os
import cv2
import time
import numpy as np

# Decide backend safely. On some ARM systems importing torch causes SIGILL and kills the process.
# To avoid that, probe torch import in a subprocess first and only import ultralytics if it succeeds.
import subprocess
import sys

BACKEND = "ultralytics"
MODEL = None
MODEL_NAMES = None

def _can_import_torch(timeout=5):
    try:
        res = subprocess.run([sys.executable, "-c", "import torch"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return res.returncode == 0
    except Exception:
        return False

if _can_import_torch():
    try:
        from ultralytics import YOLO
        MODEL = YOLO("yolov10n.pt")
        MODEL_NAMES = getattr(MODEL, "names", None)
        BACKEND = "ultralytics"
        print("Using ultralytics YOLO backend.")
    except Exception as e:
        print("Ultralytics import failed even though torch probe passed:", e)
        MODEL = None
        BACKEND = "onnx_or_mock"
else:
    print("torch import failed in a subprocess; skipping ultralytics import to avoid SIGILL.")
    MODEL = None
    BACKEND = "onnx_or_mock"

# If ultralytics wasn't loaded, try ONNX
if MODEL is None:
    onnx_path = "yolov10n.onnx"
    if os.path.exists(onnx_path):
        try:
            net = cv2.dnn.readNetFromONNX(onnx_path)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            MODEL = net
            BACKEND = "onnx"
            print("Using ONNX model via OpenCV DNN backend.")
        except Exception as e2:
            print("Failed to load ONNX model:", e2)
            MODEL = None
            BACKEND = "mock"
    else:
        MODEL = None
        BACKEND = "mock"
        print("Falling back to mock detection mode (no model).")

# Video capture helper (adjust preferred index as needed)
def open_video_capture(preferred_idx=1, max_idx=10):
    # Try preferred index first, then scan 0..max_idx for a usable device
    idxs = list(range(0, max_idx+1))
    if preferred_idx in idxs:
        idxs.remove(preferred_idx)
        idxs.insert(0, preferred_idx)

    for idx in idxs:
        cap = cv2.VideoCapture(idx)
        if cap is None or not cap.isOpened():
            # ensure proper release if partially opened
            try:
                cap.release()
            except Exception:
                pass
            continue
        # try to grab a frame to be sure it works
        ret, _ = cap.read()
        if ret:
            print(f"Opened camera at index {idx}")
            return cap
        else:
            try:
                cap.release()
            except Exception:
                pass
    print("No usable video capture device found (tried indices 0..{max_idx}).")
    return None

cap = open_video_capture(preferred_idx=1)

OBSTACLE_CLASSES = [
    "person", "bicycle", "car", "chair", "bench", "truck", "traffic light",
    "fire hydrant", "stop sign", "stairs","phone","scooter","motorcycle",
    "tree","bushes","cup","cups","bowl"  # stairs requires custom model later
]

# Grid visualization parameters
GRID_WIDTH = 800
GRID_HEIGHT = 600
GRID_CELL_SIZE = 50  # Size of each grid square
GRID_COLOR = (50, 50, 50)  # Dark gray for grid lines
TEXT_COLOR = (0, 255, 0)  # Green text
BOX_COLOR = (0, 255, 0)  # Green boxes

def create_grid_canvas():
    """Create a black canvas with a grid"""
    canvas = np.zeros((GRID_HEIGHT, GRID_WIDTH, 3), dtype=np.uint8)
    
    # Draw vertical lines
    for x in range(0, GRID_WIDTH, GRID_CELL_SIZE):
        cv2.line(canvas, (x, 0), (x, GRID_HEIGHT), GRID_COLOR, 1)
    
    # Draw horizontal lines
    for y in range(0, GRID_HEIGHT, GRID_CELL_SIZE):
        cv2.line(canvas, (0, y), (GRID_WIDTH, y), GRID_COLOR, 1)
    
    return canvas


# Default COCO names (80 classes) to use with ONNX exports if ultralytics names aren't available
COCO_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench',
    'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis',
    'snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife',
    'spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table',
    'toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
    'hair drier','toothbrush'
]


# Detection helper that returns list of detections: dicts with x1,y1,x2,y2,cls,conf
def run_detection(frame, backend=BACKEND, model=MODEL, conf_threshold=0.25, nms_threshold=0.45, input_size=640):
    h, w = frame.shape[:2]
    detections = []

    if backend == "ultralytics" and model is not None:
        # Use ultralytics model (original behavior)
        try:
            results = model(frame)
            for r in results:
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                for b in boxes:
                    x1, y1, x2, y2 = b.xyxy[0]
                    cls = int(b.cls[0])
                    conf = float(b.conf[0])
                    detections.append({
                        "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                        "cls": cls, "conf": conf
                    })
        except Exception as e:
            print("Ultralytics model inference failed:", e)

    elif backend == "onnx" and model is not None:
        # Use OpenCV DNN with ONNX export from Ultralytics (common export format)
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
        model.setInput(blob)
        outputs = model.forward()

        # Handle shapes: (1, N, 85) or (N,85)
        if outputs.ndim == 3:
            outs = outputs[0]
        else:
            outs = outputs

        boxes_xywh = []
        scores = []
        class_ids = []

        for row in outs:
            # Expect: [x_center, y_center, w, h, obj_conf, class1, class2, ...]
            if len(row) < 6:
                continue
            x_c, y_c, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            obj_conf = float(row[4])
            class_probs = row[5:]
            if len(class_probs) == 0:
                continue
            class_id = int(np.argmax(class_probs))
            class_conf = float(class_probs[class_id])
            conf = obj_conf * class_conf
            if conf < conf_threshold:
                continue

            # Scale coordinates from input_size to original frame size
            scale_x = w / input_size
            scale_y = h / input_size
            x_c *= scale_x
            y_c *= scale_y
            bw *= scale_x
            bh *= scale_y
            x1 = int(x_c - bw / 2)
            y1 = int(y_c - bh / 2)
            boxes_xywh.append([x1, y1, int(bw), int(bh)])
            scores.append(float(conf))
            class_ids.append(class_id)

        # Apply NMS
        if len(boxes_xywh) > 0:
            idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores, conf_threshold, nms_threshold)
            if len(idxs) > 0:
                for i in idxs.flatten():
                    x1, y1, bw, bh = boxes_xywh[i]
                    detections.append({
                        "x1": int(x1), "y1": int(y1), "x2": int(x1 + bw), "y2": int(y1 + bh),
                        "cls": int(class_ids[i]), "conf": float(scores[i])
                    })

    else:
        # Mock mode: return an empty array (or a synthetic detection for testing)
        # Here we add a synthetic detection to verify visualization if needed.
        # Remove or change this to return [] for a purely camera test.
        detections = []

    return detections


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Create grid canvas for this frame
    grid_canvas = create_grid_canvas()

    # Run detection using the selected backend (ultralytics / onnx / mock)
    detections = run_detection(frame)

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        cls = det["cls"]
        conf = det["conf"]

        # Draw box on main frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), BOX_COLOR, 2)

        # Get label name
        if BACKEND == "ultralytics" and MODEL is not None:
            label = MODEL_NAMES[cls] if MODEL_NAMES is not None and cls in MODEL_NAMES else str(cls)
        else:
            label = COCO_NAMES[cls] if 0 <= cls < len(COCO_NAMES) else str(cls)

        if label in OBSTACLE_CLASSES and conf > 0.4:
            cv2.putText(frame, f"This is a:{label} with:{conf:.2f} conf", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 2)
            print(f"ALERT: Object detected, this is a:{label} with:{conf:.2f} conf")
            print(f"  Location: Top-left ({int(x1)}, {int(y1)}) | Bottom-right ({int(x2)}, {int(y2)}) | Center ({int((x1+x2)/2)}, {int((y1+y2)/2)})")

            # Scale and draw on grid canvas
            grid_x1 = int((x1 / frame.shape[1]) * GRID_WIDTH)
            grid_y1 = int((y1 / frame.shape[0]) * GRID_HEIGHT)
            grid_x2 = int((x2 / frame.shape[1]) * GRID_WIDTH)
            grid_y2 = int((y2 / frame.shape[0]) * GRID_HEIGHT)

            grid_x1 = max(0, min(grid_x1, GRID_WIDTH - 1))
            grid_y1 = max(0, min(grid_y1, GRID_HEIGHT - 1))
            grid_x2 = max(0, min(grid_x2, GRID_WIDTH - 1))
            grid_y2 = max(0, min(grid_y2, GRID_HEIGHT - 1))

            cv2.rectangle(grid_canvas, (grid_x1, grid_y1), (grid_x2, grid_y2), BOX_COLOR, 2)
            cv2.putText(grid_canvas, f"{label} ({conf:.2f})", (grid_x1, grid_y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
            
    cv2.imshow("Obstacle Detection Demo", frame)
    cv2.imshow("Filtered Detections - Grid View", grid_canvas)

    # Quit on ESC or 'q' / 'Q'
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()




#if label in OBSTACLE_CLASSES and conf > 0.5:
    
