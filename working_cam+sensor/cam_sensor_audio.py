import cv2
import time
import numpy as np
import subprocess
import multiprocessing as mp
import threading
from ultralytics import YOLO
from evdev import InputDevice, categorize, ecodes

# =========================
# MODES
# =========================
MODE_NORMAL = 0
MODE_SCENARIO = 1
current_mode = MODE_NORMAL

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
    try:
        subprocess.Popen(["espeak", text])
    except:
        pass

def speak_blocking(text):
    subprocess.call(["espeak", text])

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
    
    try:
        sensor = VL53L5CXSensor(verbose=False)
    except Exception as e:
        print(f"Sensor failed to initialize: {e}")
        return

    while True:
        data = sensor.get_ranging_data()
        if data is not None:
            flat_data = data.flatten()
            with lock:
                for i in range(64):
                    shared_array[i] = flat_data[i]
        time.sleep(0.033)

# =========================
# SCENE DESCRIPTION (SAFE + LIGHT)
# =========================
def run_scene_description(frame):
    scene_active.set()
    speak_blocking("Analyzing scene")

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg = np.mean(gray)

        if avg < 50:
            desc = "Dark environment"
        elif avg > 180:
            desc = "Bright environment"
        else:
            desc = "Normal lighting"

        speak_blocking(desc)

    except:
        speak_blocking("Scene analysis failed")

    scene_active.clear()

def trigger_scene(frame):
    if not scene_active.is_set():
        threading.Thread(target=run_scene_description, args=(frame.copy(),), daemon=True).start()

def trigger_scene_global():
    global latest_frame
    if latest_frame is not None:
        trigger_scene(latest_frame)

# =========================
# MODE TOGGLE
# =========================
def toggle_mode():
    global current_mode

    if current_mode == MODE_NORMAL:
        current_mode = MODE_SCENARIO
        speak_text("Scenario mode")
    else:
        current_mode = MODE_NORMAL
        speak_text("Normal mode")

# =========================
# HEADPHONE LISTENER
# =========================
def headphone_listener():
    global last_volume_time

    dev = InputDevice('/dev/input/event5')
    print("Headphone control ready")

    for event in dev.read_loop():
        if event.type == ecodes.EV_KEY:
            key = categorize(event)

            if key.keystate == key.key_down:

                if key.keycode == 'KEY_PLAYCD':
                    trigger_scene_global()

                elif key.keycode == 'KEY_VOLUMEUP':
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

    print("Starting sensor process...")
    p = mp.Process(target=run_sensor_process, args=(shared_sensor_data, sensor_lock))
    p.daemon = True
    p.start()

    print("Loading YOLO...")
    model = YOLO("best.pt")

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Camera failed")
        exit()

    threading.Thread(target=headphone_listener, daemon=True).start()

    last_alert = 0

    print("System running...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            latest_frame = frame

            # =========================
            # GET SENSOR DATA (UNCHANGED)
            # =========================
            with sensor_lock:
                current_shared_data = shared_sensor_data[:]
            
            sensor_data = np.array(current_shared_data, dtype=np.int32).reshape((8, 8))

            # =========================
            # MODE HANDLING (NON-INVASIVE)
            # =========================

            if current_mode == MODE_NORMAL:

                # ===== YOUR ORIGINAL CODE (UNCHANGED) =====
                if not scene_active.is_set():
                    results = model(frame, verbose=False)

                    for r in results:
                        for b in r.boxes:
                            x1, y1, x2, y2 = map(int, b.xyxy[0])
                            conf = float(b.conf[0])
                            cls = int(b.cls[0])
                            label = model.names[cls]

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

                            if dist > 0 and dist < 1000:
                                if time.time() - last_alert > 1:
                                    speak_text(f"{label} ahead")
                                    last_alert = time.time()
                # =========================================

            elif current_mode == MODE_SCENARIO:

                # Lightweight safety mode (no YOLO slowdown)
                center_dist = sensor_data[3:5, 3:5].mean()

                if center_dist > 0 and center_dist < 700:
                    if time.time() - last_alert > 1:
                        speak_text("Obstacle very close")
                        last_alert = time.time()

            # =========================
            # DISPLAY (UNCHANGED)
            # =========================
            grid = draw_sensor(sensor_data)

            cv2.imshow("Camera", frame)
            cv2.imshow("Sensor", grid)

            if cv2.waitKey(1) == 27:
                break

    except KeyboardInterrupt:
        print("Exit")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        p.terminate()
        p.join()
