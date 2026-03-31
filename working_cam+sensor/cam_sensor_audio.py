import cv2
import time
import numpy as np
import subprocess
import multiprocessing as mp
from ultralytics import YOLO
from evdev import InputDevice, categorize, ecodes
import threading

DEVICE_PATH = "/dev/input/event5"

# =========================
# SAFE SPEECH (UPGRADE ONLY)
# =========================
def speak_text(text):
    try:
        subprocess.Popen(["/usr/bin/espeak", "-s", "200", "-v", "en+f3", text])
    except:
        pass

last_alert = 0
SPEECH_COOLDOWN = 1.0

def safe_speak(text):
    global last_alert
    if time.time() - last_alert > SPEECH_COOLDOWN:
        speak_text(text)
        last_alert = time.time()

# =========================
# MODES (DOES NOT TOUCH DETECTION)
# =========================
OBSTACLE_CLASSES = ["Bike","Car","Chair","Emergency Blue Phone","Exit sign",
                    "Person","Pole","Stairs","Tree","Washroom"]

MODES = {
    1: {"name": "Normal", "excluded": {"Person","Exit sign"}, "catch_unknown": False},
    2: {"name": "Everything", "excluded": set(), "catch_unknown": True},
    3: {"name": "Emergency", "excluded": set(), "catch_unknown": False},
    4: {"name": "Scene Mode", "excluded": set(), "catch_unknown": False},
}

current_mode = 1
NUM_MODES = len(MODES)

def get_active_classes():
    return set(OBSTACLE_CLASSES) - MODES[current_mode]["excluded"]

def mode_catches_unknown():
    return MODES[current_mode]["catch_unknown"]

# =========================
# HEADPHONE CONTROLS
# =========================
def headphone_listener():
    global current_mode
    dev = InputDevice(DEVICE_PATH)

    print("Headphone controls active")

    for event in dev.read_loop():
        if event.type == ecodes.EV_KEY:
            key_event = categorize(event)

            if key_event.keystate == 1:
                key = key_event.keycode

                if key == "KEY_VOLUMEUP":
                    current_mode = (current_mode % NUM_MODES) + 1
                    safe_speak(MODES[current_mode]["name"])

                elif key == "KEY_VOLUMEDOWN":
                    current_mode = (current_mode - 2) % NUM_MODES + 1
                    safe_speak(MODES[current_mode]["name"])

# =========================
# SCENE MODE (OPTIONAL)
# =========================
def describe_scene(labels):
    if len(labels) == 0:
        safe_speak("No major objects detected")
        return

    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1

    parts = []
    for k, v in counts.items():
        parts.append(f"a {k}" if v == 1 else f"{v} {k}s")

    safe_speak("I see " + ", ".join(parts))

# =========================
# SENSOR DRAW (UNCHANGED)
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
            flat_data = data.flatten()
            with lock:
                for i in range(64):
                    shared_array[i] = flat_data[i]

        time.sleep(0.033)

# =========================
# MAIN (UNCHANGED CORE)
# =========================
if __name__ == '__main__':

    shared_sensor_data = mp.Array('i', 64)
    sensor_lock = mp.Lock()

    p = mp.Process(target=run_sensor_process, args=(shared_sensor_data, sensor_lock))
    p.daemon = True
    p.start()

    threading.Thread(target=headphone_listener, daemon=True).start()

    model = YOLO("best.pt")

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Camera failed")
        exit()

    print("System running")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            with sensor_lock:
                sensor_data = np.array(shared_sensor_data[:], dtype=np.int32).reshape((8, 8))

            results = model(frame, verbose=False)

            scene_labels = []

            for r in results:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    conf = float(b.conf[0])
                    if conf < 0.5:
                        continue

                    cls = int(b.cls[0])
                    label = model.names[cls]

                    # MODE FILTER (speech only)
                    active = get_active_classes()
                    if label not in active:
                        if mode_catches_unknown():
                            label = "object"
                        else:
                            continue

                    scene_labels.append(label)

                    cx = int((x1 + x2) / 2 / frame.shape[1] * 8)
                    cy = int((y1 + y2) / 2 / frame.shape[0] * 8)

                    cx = max(0, min(7, cx))
                    cy = max(0, min(7, cy))

                    dist = sensor_data[cy, cx]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {dist}mm",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

                    # 🔊 ONLY CHANGE: safe_speak instead of speak_text
                    if dist > 0 and dist < 1000:
                        safe_speak(f"{label} ahead")

            if current_mode == 4:
                describe_scene(scene_labels)

            cv2.imshow("Camera", frame)
            cv2.imshow("Sensor", draw_sensor(sensor_data))

            if cv2.waitKey(1) == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        p.terminate()
        p.join()
