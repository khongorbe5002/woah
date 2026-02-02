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
# Use a smaller default capture resolution to reduce decoder/CPU overhead when necessary
def open_video_capture(preferred_idx=1, max_idx=10, request_width=640, request_height=480, use_mjpg=True):
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

        # Try to set desired properties (may silently fail if unsupported)
        try:
            if use_mjpg:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, request_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, request_height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            # Try disabling autofocus if supported
            try:
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            except Exception:
                pass
        except Exception:
            pass

        # try to grab a frame to be sure it works
        ret, _ = cap.read()
        if ret:
            # log the actual capture properties
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Opened camera at index {idx} (reported size: {int(w)}x{int(h)} @ {fps:.1f} fps)")
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

# Camera & preprocessing tweaks
REQUEST_CAMERA_WIDTH = 640
REQUEST_CAMERA_HEIGHT = 480
USE_MJPG = True
# Software sharpening: set to True to apply Unsharp Mask fallback in case hardware focus is bad
SHARPEN = True
SHARPEN_ALPHA = 1.5  # weight for original image in unsharp mask
SHARPEN_BETA = -0.5  # weight for blurred image in unsharp mask
SHARPEN_SIGMA = 3  # Gaussian blur sigma

# Performance / filtering controls (cleaned defaults)
# Input size for ONNX: many exports expect 640; keep 640 for compatibility.
INPUT_SIZE = 640
# Process every Nth frame to reduce CPU load (increase to further reduce inferences/prints)
PROCESS_EVERY_N = 6
# How often to update the GUI windows (1 = every frame, increase to throttle GUI updates)
SHOW_EVERY_N = 1
# Show running FPS + average inference ms on the camera window
SHOW_PERF_OVERLAY = True

# Obstacle alert settings (reduce print spam)
OBSTACLE_ALERT_CONF = 0.5  # minimum confidence to consider printing an alert
ALERT_PRINT_COOLDOWN = 3.0  # seconds between printing the same label alert

# Performance: allow OpenCV to use multiple CPU threads
cv2.setNumThreads(max(1, os.cpu_count() - 1))
try:
    cv2.ocl.setUseOpenCL(True)
except Exception:
    pass
# Performance: allow OpenCV to use multiple CPU threads
cv2.setNumThreads(max(1, os.cpu_count() - 1))
try:
    cv2.ocl.setUseOpenCL(True)
except Exception:
    pass

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
        # Use OpenCV DNN with ONNX export (supports classic YOLO format or separate 'logits'+'pred_boxes')
        used_input_size = input_size
        try:
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
            model.setInput(blob)
            outputs = model.forward()
        except Exception:
            # Some ONNX exports rely on specific input sizes. Retry with 640 if the requested smaller size fails.
            outputs = None
            if input_size != 640:
                try:
                    used_input_size = 640
                    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (used_input_size, used_input_size), swapRB=True, crop=False)
                    model.setInput(blob)
                    outputs = model.forward()
                    # warn once
                    print("Warning: model failed with input_size", input_size, "; falling back to 640 for this frame")
                except Exception:
                    outputs = None

        boxes_xywh = []
        scores = []
        class_ids = []

        # Case A: model returns rows containing [x_c, y_c, w, h, obj_conf, class_probs...]
        parsed = False
        if outputs is not None:
            if outputs.ndim == 3:
                outs = outputs[0]
            else:
                outs = outputs

            # If rows have at least 6 entries, assume YOLO-like output
            if outs.shape[1] >= 6:
                for row in outs:
                    if row.shape[0] < 6:
                        continue
                    x_c, y_c, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    obj_conf = float(row[4])
                    class_probs = row[5:]
                    if class_probs.size == 0:
                        continue
                    class_id = int(np.argmax(class_probs))
                    class_conf = float(class_probs[class_id])
                    conf = obj_conf * class_conf
                    if conf < conf_threshold:
                        continue

                    # Scale coordinates from used_input_size to original frame size
                    scale_x = w / used_input_size
                    scale_y = h / used_input_size
                    x_c *= scale_x
                    y_c *= scale_y
                    bw *= scale_x
                    bh *= scale_y
                    x1 = int(x_c - bw / 2)
                    y1 = int(y_c - bh / 2)
                    boxes_xywh.append([x1, y1, int(bw), int(bh)])
                    scores.append(float(conf))
                    class_ids.append(class_id)

                parsed = True

        # Case B: model exposes separate 'logits' and 'pred_boxes' outputs (e.g., yolo26n)
        if not parsed:
            try:
                logits = model.forward('logits')  # shape: (1, N, C)
                pred_boxes = model.forward('pred_boxes')  # shape: (1, N, 4)
                logits = np.array(logits)
                pred_boxes = np.array(pred_boxes)
                if logits.ndim == 3 and pred_boxes.ndim == 3:
                    probs = np.exp(logits - np.max(logits, axis=2, keepdims=True))
                    probs = probs / probs.sum(axis=2, keepdims=True)
                    arr_logits = probs[0]
                    arr_boxes = pred_boxes[0]
                    for i in range(arr_boxes.shape[0]):
                        class_id = int(np.argmax(arr_logits[i]))
                        class_conf = float(arr_logits[i, class_id])
                        conf = class_conf
                        if conf < conf_threshold:
                            continue

                        x_c, y_c, bw, bh = map(float, arr_boxes[i][:4])
                        # pred_boxes are normalized [0,1], scale to frame
                        x_c_px = x_c * w
                        y_c_px = y_c * h
                        bw_px = bw * w
                        bh_px = bh * h
                        x1 = int(x_c_px - bw_px / 2)
                        y1 = int(y_c_px - bh_px / 2)
                        boxes_xywh.append([x1, y1, int(bw_px), int(bh_px)])
                        scores.append(float(conf))
                        class_ids.append(class_id)
            except Exception:
                # Could not parse specialized outputs; leave detections empty
                pass

        # Apply NMS if boxes found
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


# Performance counters
frame_counter = 0
fps_last_time = time.time()
frames_since_last = 0
fps = 0.0
avg_infer_ms = 0.0
inference_count = 0
last_detections = []
last_alert_times = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    frames_since_last += 1

    # Apply optional software sharpening if hardware focus is poor
    if SHARPEN:
        try:
            blur = cv2.GaussianBlur(frame, (0, 0), SHARPEN_SIGMA)
            frame = cv2.addWeighted(frame, SHARPEN_ALPHA, blur, SHARPEN_BETA, 0)
        except Exception:
            pass

    # Create grid canvas for this frame
    grid_canvas = create_grid_canvas()

    # Decide whether to process this frame (skip some frames to save CPU)
    process_this = (frame_counter % PROCESS_EVERY_N == 0)

    infer_ms = 0.0
    detections = []
    if process_this:
        t0 = time.time()
        detections = run_detection(frame, input_size=INPUT_SIZE)
        infer_ms = (time.time() - t0) * 1000.0
        inference_count += 1
        # running average for inference ms
        avg_infer_ms = ((avg_infer_ms * (inference_count - 1)) + infer_ms) / inference_count if inference_count > 0 else infer_ms
        last_detections = detections
    else:
        # reuse last detections for smoother display when skipping processing
        detections = last_detections

    # Update FPS once per second
    if time.time() - fps_last_time >= 1.0:
        elapsed = time.time() - fps_last_time
        fps = frames_since_last / elapsed if elapsed > 0 else 0.0
        print(f"FPS: {fps:.2f}, avg_infer_ms: {avg_infer_ms:.1f} ms, process_every_n: {PROCESS_EVERY_N}, input_size: {INPUT_SIZE}")
        frames_since_last = 0
        fps_last_time = time.time()

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        cls = det["cls"]
        conf = det["conf"]

        # Get label name
        if BACKEND == "ultralytics" and MODEL is not None:
            label = MODEL_NAMES[cls] if MODEL_NAMES is not None and cls in MODEL_NAMES else str(cls)
        else:
            label = COCO_NAMES[cls] if 0 <= cls < len(COCO_NAMES) else str(cls)

        # Decide box color (highlight obstacles)
        draw_color = BOX_COLOR
        if label in OBSTACLE_CLASSES and conf > OBSTACLE_ALERT_CONF:
            draw_color = (0, 0, 255)  # red for obstacles

        # Draw box on main frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), draw_color, 2)

        # Draw label and confidence on the camera window (always)
        text = f"{label} {conf:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x = max(0, min(int(x1), frame.shape[1] - text_w - 1))
        text_y = max(text_h + 2, int(y1) - 6)
        # background rectangle for text for visibility
        cv2.rectangle(frame, (text_x, text_y - text_h - baseline), (text_x + text_w, text_y + baseline), draw_color, -1)
        cv2.putText(frame, text, (text_x, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # If detection is an obstacle above threshold, print alert and draw on grid (with cooldown)
        if label in OBSTACLE_CLASSES and conf > OBSTACLE_ALERT_CONF:
            now = time.time()
            last = last_alert_times.get(label, 0)
            if now - last > ALERT_PRINT_COOLDOWN:
                print(f"ALERT: Object detected, this is a:{label} with:{conf:.2f} conf")
                print(f"  Location: Top-left ({int(x1)}, {int(y1)}) | Bottom-right ({int(x2)}, {int(y2)}) | Center ({int((x1+x2)/2)}, {int((y1+y2)/2)})")
                last_alert_times[label] = now

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
            
# Overlay performance info
        if SHOW_PERF_OVERLAY:
            perf_text = f"FPS:{fps:.1f} INF:{avg_infer_ms:.1f}ms"
            cv2.putText(frame, perf_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Obstacle Detection Demo", frame)
    cv2.imshow("Filtered Detections - Grid View", grid_canvas)

    # Quit on ESC or 'q' / 'Q'
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()




#if label in OBSTACLE_CLASSES and conf > 0.5:
    
