from ultralytics import YOLO
import cv2
import time
import threading
import numpy as np
import subprocess

from working_cam_sensor import VL53L5CXSensor


def speak_text(text):
    try:
        subprocess.Popen(
            ["espeak-ng", "-s", "160", "-v", "en", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except:
        pass


OBSTACLE_CLASSES = [
    "Bike", "Car", "Chair", "Emergency Blue Phone",
    "Exit sign", "Person", "Pole", "Stairs", "Tree", "Washroom",
]

CLOSE_THRESHOLD_MM = 1000
ALERT_REMINDER_SECONDS = 1.0
TRACKED_OBJECT_MOVE_THRESHOLD_PX = 80
TRACKED_OBJECT_MAX_AGE = 10.0
SENSOR_STALE_TIMEOUT = 0.5


tracked_objects = []
prev_close_cells = set()


def get_direction(cx, cy, w, h):
    third_w = w / 3
    third_h = h / 3

    if cx < third_w:
        horiz = "left"
    elif cx < 2 * third_w:
        horiz = "center"
    else:
        horiz = "right"

    if cy < third_h:
        vert = "upper"
    elif cy < 2 * third_h:
        vert = "center"
    else:
        vert = "bottom"

    if vert == "center" and horiz == "center":
        return "center"
    if vert == "center":
        return horiz
    if horiz == "center":
        return vert
    return f"{vert} {horiz}"


def dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5


def find_obj(label, center):
    for o in tracked_objects:
        if o["label"] == label and dist(center, o["center"]) < TRACKED_OBJECT_MOVE_THRESHOLD_PX:
            return o
    return None


def cleanup(now):
    global tracked_objects
    tracked_objects = [o for o in tracked_objects if now - o["last_seen"] < TRACKED_OBJECT_MAX_AGE]


sensor_lock = threading.Lock()
last_sensor = np.zeros((8, 8), dtype=np.int32)
last_update = time.time()
stop_event = threading.Event()


def normalize(raw):
    if raw is None:
        return None
    try:
        arr = np.asarray(raw)
    except:
        return None
    if arr.shape == (8, 8):
        return arr.astype(np.int32)
    if arr.size == 64:
        return arr.reshape((8, 8)).astype(np.int32)
    flat = arr.flatten()
    if flat.size < 64:
        flat = np.concatenate([flat, np.zeros(64-flat.size)])
    else:
        flat = flat[:64]
    return flat.reshape((8, 8)).astype(np.int32)


def sensor_thread():
    global last_sensor, last_update
    while not stop_event.is_set():
        if sensor is None:
            time.sleep(0.1)
            continue
        data = sensor.get_ranging_data()
        if data is None:
            time.sleep(0.005)
            continue
        arr = normalize(data)
        if arr is None:
            continue
        with sensor_lock:
            last_sensor = arr
        last_update = time.time()


def map_bbox(x1, y1, x2, y2, h, w):
    c1 = max(0, min(int((x1/w)*8), 7))
    r1 = max(0, min(int((y1/h)*8), 7))
    c2 = max(0, min(int((x2/w)*8), 7))
    r2 = max(0, min(int((y2/h)*8), 7))
    return [(r, c) for r in range(r1, r2+1) for c in range(c1, c2+1)]


def check_close(sensor_data, cells):
    vals = [sensor_data[r, c] for r, c in cells if sensor_data[r, c] > 0]
    if not vals:
        return False, None
    m = min(vals)
    return m < CLOSE_THRESHOLD_MM, m


model = YOLO("best.pt", verbose=False)
model.fuse()

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if cap.isOpened():
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
else:
    cap = None


try:
    sensor = VL53L5CXSensor(verbose=False)
except Exception as e:
    print(f"Sensor error: {e}")
    sensor = None


threading.Thread(target=sensor_thread, daemon=True).start()

speak_text("System ready")


try:
    while True:
        if cap is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            ret, frame = cap.read()
            if not ret:
                continue

        with sensor_lock:
            sensor_data = last_sensor.copy()
            t = last_update

        if time.time() - t > SENSOR_STALE_TIMEOUT:
            sensor_data = np.zeros((8, 8), dtype=np.int32)

        results = model(frame, stream=True, verbose=False)

        audio_sent = False
        now = time.time()
        cleanup(now)
        current_cells = set()

        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0]
                cls = int(b.cls[0])
                label = model.names[cls]

                if label not in OBSTACLE_CLASSES:
                    continue

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                cells = map_bbox(x1, y1, x2, y2, frame.shape[0], frame.shape[1])
                cells = [(r, c) for r, c in cells if r < 4]

                close, dist_mm = check_close(sensor_data, cells)

                for r0, c0 in cells:
                    if 0 < sensor_data[r0, c0] < CLOSE_THRESHOLD_MM:
                        current_cells.add((r0, c0))

                confirmed = close and any(
                    (r0, c0) in prev_close_cells
                    for r0, c0 in cells
                    if 0 < sensor_data[r0, c0] < CLOSE_THRESHOLD_MM
                )

                if confirmed:
                    obj = find_obj(label, (cx, cy))
                    alert = False

                    if obj is None:
                        obj = {"label": label, "center": (cx, cy), "last_seen": now, "last_alert": now}
                        tracked_objects.append(obj)
                        alert = True
                    else:
                        moved = dist((cx, cy), obj["center"])
                        obj["center"] = (cx, cy)
                        obj["last_seen"] = now
                        if now - obj["last_alert"] > ALERT_REMINDER_SECONDS and moved < TRACKED_OBJECT_MOVE_THRESHOLD_PX:
                            obj["last_alert"] = now
                            alert = True

                    if alert and not audio_sent:
                        direction = get_direction(cx, cy, frame.shape[1], frame.shape[0])
                        print(f"{label} {dist_mm}mm {direction}")
                        speak_text(f"{label} {direction}")
                        audio_sent = True

        prev_close_cells = current_cells

except KeyboardInterrupt:
    pass


stop_event.set()

if cap:
    cap.release()

if sensor:
    sensor.close()
