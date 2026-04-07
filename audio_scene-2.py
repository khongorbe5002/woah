import os
import cv2
import time
import numpy as np
import subprocess
import multiprocessing as mp
import threading
import torch
import selectors
from evdev import list_devices, InputDevice, categorize, ecodes
from ultralytics import YOLO

# =========================
# CPU LIMIT (CRITICAL)
# =========================
torch.set_num_threads(2)

# =========================
# MODES & CLASSES
# =========================
MODE_NORMAL = 0
MODE_EVERYTHING = 1
MODE_EMERGENCY = 2
current_mode = MODE_NORMAL

# The master list of objects your YOLO model knows how to detect
OBSTACLE_CLASSES = {
    "Bike", "Bottle", "Branch", "Chair", "Emergency Blue Phone",
    "Exit Sign", "Garbage Can", "Person", "Phone", "Pole",
    "Push to Open Button", "Sanitizer", "Stairs", "Tree",
    "Vehicle", "Washroom", "Water Fountain"
}

# =========================
# GLOBALS
# =========================
scene_active = threading.Event()
latest_frame = None

# BLIP globals
blip_model = None
blip_processor = None

# =========================
# TTS
# =========================
def speak_text(text):
    subprocess.Popen(["espeak", text])

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
# SENSOR PROCESS 
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
# SCENE DESCRIPTION (FAST)
# =========================
def run_scene_description(frame):
    global blip_model, blip_processor
    from PIL import Image

    scene_active.set()
    speak_blocking("Analyzing scene")

    try:
        if frame is None:
            speak_blocking("No image")
            scene_active.clear()
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 🔥 smaller image = faster
        image = Image.fromarray(rgb).convert("RGB").resize((160, 160))

        inputs = blip_processor(image, return_tensors="pt")

        with torch.no_grad():
            out = blip_model.generate(**inputs, max_new_tokens=15)

        caption = blip_processor.decode(out[0], skip_special_tokens=True)

        print("Scene:", caption)
        speak_blocking(caption)

    except Exception as e:
        print("Scene error:", e)
        speak_blocking("Scene failed")

    scene_active.clear()

def trigger_scene(frame):
    if not scene_active.is_set():
        threading.Thread(
            target=run_scene_description,
            args=(frame.copy(),),
            daemon=True
        ).start()

def trigger_scene_global():
    global latest_frame
    if latest_frame is not None:
        trigger_scene(latest_frame)

# =========================
# MODE TOGGLES
# =========================
def toggle_everything_mode():
    global current_mode
    if current_mode == MODE_EVERYTHING:
        current_mode = MODE_NORMAL
        print("\n--- MODE CHANGED: NORMAL MODE ---")
        speak_text("Normal mode")
    else:
        current_mode = MODE_EVERYTHING
        print("\n--- MODE CHANGED: EVERYTHING MODE ---")
        speak_text("Everything mode")

def toggle_emergency_mode():
    global current_mode
    if current_mode == MODE_EMERGENCY:
        current_mode = MODE_NORMAL
        print("\n--- MODE CHANGED: NORMAL MODE ---")
        speak_text("Normal mode")
    else:
        current_mode = MODE_EMERGENCY
        print("\n--- MODE CHANGED: EMERGENCY MODE ---")
        speak_text("Emergency mode")

# =========================
# HEADPHONE LISTENER
# =========================
def headphone_listener():
    paths = list_devices()
    sel = selectors.DefaultSelector()
    devices = []

    for path in paths:
        try:
            dev = InputDevice(path)
            sel.register(dev, selectors.EVENT_READ)
            devices.append(dev)
        except PermissionError:
            pass

    print(f"Headphone listener actively monitoring {len(devices)} input streams...")

    try:
        while True:
            for key, mask in sel.select():
                dev = key.fileobj
                for event in dev.read():
                    if event.type == ecodes.EV_KEY:
                        k = categorize(event)

                        if k.keystate == k.key_down:
                            print(f"BUTTON DETECTED: {k.keycode} on {dev.path}")

                            # ▶️ Scene Analysis
                            if k.keycode in ['KEY_PLAYCD', 'KEY_PLAYPAUSE']:
                                trigger_scene_global()

                            # ⏭️ Toggle Everything Mode
                            elif k.keycode == 'KEY_NEXTSONG':
                                toggle_everything_mode()

                            # ⏮️ Toggle Emergency Mode
                            elif k.keycode == 'KEY_PREVIOUSSONG':
                                toggle_emergency_mode()

    except Exception as e:
        print(f"Listener shut down: {e}")

# =========================
# MAIN
# =========================
if __name__ == '__main__':
    
    # Sensor multiprocessing
    shared_sensor_data = mp.Array('i', 64)
    sensor_lock = mp.Lock()

    print("Starting sensor process...")
    p = mp.Process(target=run_sensor_process, args=(shared_sensor_data, sensor_lock))
    p.daemon = True
    p.start()

    # Load YOLO
    print("Loading YOLO...")
    model = YOLO("best6_ncnn_model")

    # 🔥 Load BLIP ONCE (FIXES DELAY)
    print("Loading BLIP (one-time)...")
    from transformers import BlipProcessor, BlipForConditionalGeneration
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to("cpu")
    blip_model.eval()
    print("BLIP ready")

    # Camera
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Camera failed")
        exit()

    # Headphone control thread
    threading.Thread(target=headphone_listener, daemon=True).start()

    last_alert = 0

    print("\nSystem running...")
    print("--- STARTING IN: NORMAL MODE ---")
    speak_text("System ready. Normal mode.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # =========================
            # ROTATE CAMERA
            # =========================
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            latest_frame = frame

            # Get sensor data
            with sensor_lock:
                current_shared_data = shared_sensor_data[:]
            
            sensor_data = np.array(current_shared_data, dtype=np.int32).reshape((8, 8))

            # =========================
            # APPLY MODE RULES
            # =========================
            if current_mode == MODE_NORMAL:
                active_classes = OBSTACLE_CLASSES - {"Person"}
                catch_unknown = False
            elif current_mode == MODE_EVERYTHING:
                active_classes = OBSTACLE_CLASSES
                catch_unknown = True
            elif current_mode == MODE_EMERGENCY:
                active_classes = OBSTACLE_CLASSES
                catch_unknown = False

            # =========================
            # UNIFIED VISION & SENSOR LOOP
            # =========================
            if not scene_active.is_set():
                results = model(frame, verbose=False)
                yolo_alert_triggered = False 

                for r in results:
                    for b in r.boxes:
                        cls = int(b.cls[0])
                        label = model.names[cls]

                        # Ignore objects not active in this mode
                        if label not in active_classes:
                            continue

                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        conf = float(b.conf[0])

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

                        # If YOLO sees an active object within 1 meter
                        if dist > 0 and dist < 1000:
                            yolo_alert_triggered = True
                            if time.time() - last_alert > 1:
                                speak_text(f"{label} ahead")
                                last_alert = time.time()

                # --- UNKNOWN OBJECT FALLBACK ---
                # Runs ONLY if catch_unknown is True (Everything Mode) 
                # and YOLO didn't already alert you to something close
                if catch_unknown and not yolo_alert_triggered:
                    
                    # Grab the center 4x4 grid of the sensor
                    center_patch = sensor_data[2:6, 2:6]
                    valid_dists = center_patch[center_patch > 0]
                    
                    if len(valid_dists) > 0:
                        closest_unknown = np.min(valid_dists)
                        
                        if closest_unknown < 1000:
                            if time.time() - last_alert > 1:
                                speak_text("Unknown object ahead")
                                last_alert = time.time()

            # =========================
            # DISPLAY
            # =========================
            grid = draw_sensor(sensor_data)

            # Draw the current mode on the top-left of the camera feed
            if current_mode == MODE_NORMAL:
                mode_text = "MODE: NORMAL"
            elif current_mode == MODE_EVERYTHING:
                mode_text = "MODE: EVERYTHING"
            else:
                mode_text = "MODE: EMERGENCY"
                
            cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Camera", frame)
            cv2.imshow("Sensor", grid)

            if cv2.waitKey(1) == 27:
                break

    except KeyboardInterrupt:
        print("\nExit")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        p.terminate()
        p.join()