import cv2
import time
import numpy as np
import subprocess
import multiprocessing as mp
import threading
from ultralytics import YOLO
from evdev import InputDevice, categorize, ecodes

# =========================
# MODES + CLASSES
# =========================
OBSTACLE_CLASSES = [
    "Bike", "Bottle", "Branch", "Chair", "Emergency Blue Phone",
    "Exit Sign", "Garbage Can", "Person", "Phone", "Pole",
    "Push to Open Button", "Sanitizer", "Stairs", "Tree",
    "Vehicle", "Washroom", "Water Fountain",
]

MODES = {
    1: {"name": "Normal", "spoken": "Normal mode",
        "excluded": {"Person", "Emergency Blue Phone", "Exit Sign"},
        "catch_unknown": False},
    2: {"name": "Everything", "spoken": "Everything mode",
        "excluded": set(), "catch_unknown": True},
    3: {"name": "Emergency", "spoken": "Emergency mode",
        "excluded": set(), "catch_unknown": False},
}

NUM_MODES = len(MODES)
current_mode = 1
_mode_lock = threading.Lock()

# =========================
# GLOBALS
# =========================
last_volume_time = 0
DOUBLE_PRESS_WINDOW = 0.5

# =========================
# TTS
# =========================
def speak_text(text):
    subprocess.Popen(["espeak", text])

# =========================
# SENSOR VISUALIZATION
# =========================
def draw_sensor(sensor_data):
    size = 400
    cell = size // 8
    img = np.zeros((size, size, 3), dtype=np.uint8)

    for r in range(8):
        for c in range(8):
            d = sensor_data[r, c]
            if d == 0:
                color = (50, 50, 50)
            else:
                v = min(d, 3000) / 3000
                color = (int(255 * v), int(255 * (1 - v)), 0)

            x1, y1 = c * cell, r * cell
            x2, y2 = x1 + cell, y1 + cell
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    return img

# =========================
# SENSOR PROCESS (UNCHANGED)
# =========================
def run_sensor_process(shared_array, lock):
    from working_cam_sensor import VL53L5CXSensor
    
    sensor = VL53L5CXSensor(verbose=False)

    while True:
        data = sensor.get_ranging_data()
        if data is not None:
            flat = data.flatten()
            with lock:
                for i in range(64):
                    shared_array[i] = flat[i]
        time.sleep(0.033)

# =========================
# MODE TOGGLE
# =========================
def toggle_mode():
    global current_mode

    with _mode_lock:
        current_mode += 1
        if current_mode > NUM_MODES:
            current_mode = 1

        speak_text(MODES[current_mode]["spoken"])
        print("Mode:", MODES[current_mode]["name"])

# =========================
# HEADPHONE LISTENER
# =========================
def headphone_listener():
    global last_volume_time
    dev = InputDevice('/dev/input/event9')

    for event in dev.read_loop():
        if event.type == ecodes.EV_KEY:
            key = categorize(event)

            if key.keystate == key.key_down:
                if key.keycode in ['KEY_VOLUMEUP', 'KEY_VOLUME_UP']:
                    now = time.time()

                    if now - last_volume_time < DOUBLE_PRESS_WINDOW:
                        toggle_mode()
                        last_volume_time = 0
                    else:
                        last_volume_time = now

# =========================
# MAIN
# =========================
if __name__ == '__main__':
    
    shared_sensor_data = mp.Array('i', 64)
    sensor_lock = mp.Lock()

    p = mp.Process(target=run_sensor_process, args=(shared_sensor_data, sensor_lock))
    p.daemon = True
    p.start()

    model = YOLO("best6.pt")

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Camera failed")
        exit()

    threading.Thread(target=headphone_listener, daemon=True).start()

    last_alert = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with sensor_lock:
            sensor_data = np.array(shared_sensor_data[:], dtype=np.int32).reshape((8, 8))

        results = model(frame, verbose=False)

        with _mode_lock:
            mode_cfg = MODES[current_mode]

        # 🔥 Track best object
        best_distance = 99999
        best_label = None
        best_direction = None

        for r in results:
            for b in r.boxes:

                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cls = int(b.cls[0])
                label = model.names[cls]

                if label not in OBSTACLE_CLASSES:
                    continue

                if label in mode_cfg["excluded"]:
                    continue

                # =========================
                # BBOX → SENSOR GRID
                # =========================
                norm_x1 = x1 / frame.shape[1]
                norm_y1 = y1 / frame.shape[0]
                norm_x2 = x2 / frame.shape[1]
                norm_y2 = y2 / frame.shape[0]

                col1 = int(norm_x1 * 8)
                row1 = int(norm_y1 * 8)
                col2 = int(norm_x2 * 8)
                row2 = int(norm_y2 * 8)

                col1 = max(0, min(7, col1))
                col2 = max(0, min(7, col2))
                row1 = max(0, min(7, row1))
                row2 = max(0, min(7, row2))

                distances = []

                for rr in range(row1, row2 + 1):
                    for cc in range(col1, col2 + 1):
                        d = sensor_data[rr, cc]
                        if d > 0:
                            distances.append(d)

                if len(distances) == 0:
                    continue

                dist = min(distances)

                # =========================
                # CAMERA-BASED DIRECTION
                # =========================
                cx_pixel = int((x1 + x2) / 2)
                cx = int(cx_pixel / frame.shape[1] * 8)

                if cx <= 2:
                    direction = "left"
                elif cx >= 5:
                    direction = "right"
                else:
                    direction = "ahead"

                # Track closest object
                if dist < best_distance:
                    best_distance = dist
                    best_label = label
                    best_direction = direction

                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{label} {dist}mm",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,255,0), 2)

        # =========================
        # SPEAK ONLY BEST OBJECT
        # =========================
        if best_label is not None:
            if best_distance < 1200 and time.time() - last_alert > 1.5:
                speak_text(f"{best_label} {best_direction}")
                last_alert = time.time()

        # UNKNOWN (Mode 2)
        if mode_cfg["catch_unknown"]:
            center = sensor_data[3:5, 3:5].mean()
            if center > 0 and center < 800:
                if time.time() - last_alert > 1:
                    speak_text("Unknown obstacle ahead")
                    last_alert = time.time()

        # DISPLAY
        grid = draw_sensor(sensor_data)
        cv2.imshow("Camera", frame)
        cv2.imshow("Sensor", grid)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    p.terminate()
    p.join()
