import os
os.environ["HF_HOME"] = "/home/pi/hf_cache"

import cv2
import time
import numpy as np
import subprocess
import multiprocessing as mp
import threading
import torch
from ultralytics import YOLO
from evdev import InputDevice, categorize, ecodes

torch.set_num_threads(1)

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
scene_active = threading.Event()
latest_frame = None
last_volume_time = 0
DOUBLE_PRESS_WINDOW = 0.5

# =========================
# TTS
# =========================
def speak_text(text):
    subprocess.Popen(["espeak", text])

# =========================
# SENSOR PROCESS (CORRECT)
# =========================
def run_sensor_process(shared_array, lock):
    from working_cam_sensor import VL53L5CXSensor
    
    sensor = VL53L5CXSensor(verbose=False)
    print("Sensor initialized")

    while True:
        try:
            data = sensor.get_ranging_data()

            if data is not None and data.shape == (8, 8):

                # ✅ Correct orientation
                data = np.rot90(data, 2)

                # ✅ Remove invalid readings
                data = np.where((data == 0) | (data > 3000), 0, data)

                flat = data.flatten()

                with lock:
                    for i in range(64):
                        shared_array[i] = int(flat[i])

        except Exception as e:
            print("Sensor error:", e)

        time.sleep(0.02)

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

        results = model(frame, imgsz=320, verbose=False)

        with _mode_lock:
            mode_cfg = MODES[current_mode]

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
                # CAMERA → SENSOR MAPPING
                # =========================
                cx = int(((x1 + x2) / 2) / frame.shape[1] * 8)
                cy = int(((y1 + y2) / 2) / frame.shape[0] * 8)

                cx = max(0, min(7, cx))
                cy = max(0, min(7, cy))

                # Flip X if needed
                cx = 7 - cx

                # =========================
                # PROPER DISTANCE
                # =========================
                region = sensor_data[max(0,cy-1):min(8,cy+2),
                                     max(0,cx-1):min(8,cx+2)]

                valid = region[region > 0]

                if len(valid) == 0:
                    continue

                dist = int(np.min(valid))  # closest object

                # Filter unrealistic values
                if dist < 150 or dist > 2500:
                    continue

                # =========================
                # DIRECTION (CORRECTED)
                # =========================
                if cx <= 2:
                    direction = "left"
                elif cx >= 5:
                    direction = "right"
                else:
                    direction = "ahead"

                # =========================
                # DISPLAY + SPEECH
                # =========================
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{label} {dist}mm",
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,255,0), 2)

                if dist < 1200 and time.time() - last_alert > 1:
                    speak_text(f"{label} {direction}")
                    last_alert = time.time()

        # =========================
        # UNKNOWN DETECTION (MODE 2)
        # =========================
        if mode_cfg["catch_unknown"]:
            center = sensor_data[3:5, 3:5]
            valid = center[center > 0]

            if len(valid) > 0:
                d = int(np.min(valid))
                if d < 800 and time.time() - last_alert > 1:
                    speak_text("Unknown obstacle ahead")
                    last_alert = time.time()

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    p.terminate()
    p.join()
