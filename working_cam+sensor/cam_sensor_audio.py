import cv2
import time
import numpy as np
import subprocess
import multiprocessing as mp
import threading
import queue
import torch
from ultralytics import YOLO
from evdev import InputDevice, categorize, ecodes
from transformers import BlipProcessor, BlipForConditionalGeneration

# =========================
# CONFIG
# =========================
torch.set_num_threads(2)

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
# TTS SYSTEM
# =========================
tts_queue = queue.Queue()

def tts_worker():
    while True:
        text = tts_queue.get()
        subprocess.run(["espeak", text])

threading.Thread(target=tts_worker, daemon=True).start()

def speak_text(text):
    while not tts_queue.empty():
        try:
            tts_queue.get_nowait()
        except:
            break
    tts_queue.put(text)

def speak_blocking(text):
    subprocess.call(["espeak", text])

# =========================
# GLOBALS
# =========================
last_press_time = 0
DOUBLE_PRESS_WINDOW = 0.5
scene_active = threading.Event()

last_frame = None
last_sensor = None

# =========================
# BLIP (SCENE AI)
# =========================
blip_model = None
blip_processor = None

def run_scene_description(frame):
    global blip_model, blip_processor

    from PIL import Image

    scene_active.set()
    speak_blocking("Analyzing scene")

    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize((160, 160))

        inputs = blip_processor(image, return_tensors="pt")

        with torch.no_grad():
            out = blip_model.generate(**inputs, max_new_tokens=15)

        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        print("Scene:", caption)

        speak_blocking(caption)

    except Exception as e:
        print("BLIP error:", e)
        speak_blocking("Scene failed")

    scene_active.clear()

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
# MODE CONFIG
# =========================
MODE_ORDER = [1, 3, 2]   # Normal → Emergency → Everything
current_mode_index = 0
current_mode = MODE_ORDER[current_mode_index]
_mode_lock = threading.Lock()

# =========================
# MODE TOGGLE
# =========================
def toggle_mode():
    global current_mode_index, current_mode

    with _mode_lock:
        current_mode_index = (current_mode_index + 1) % len(MODE_ORDER)
        current_mode = MODE_ORDER[current_mode_index]
        mode_name = MODES[current_mode]["name"]

    print(f"[MODE SWITCH] {mode_name}")
    speak_text(f"{mode_name} mode")
# =========================
# HEADPHONE LISTENER
# =========================
def headphone_listener():
    global last_press_time

    dev = InputDevice('/dev/input/event5')
    print("Headphone ready:", dev)

    for event in dev.read_loop():
        if event.type == ecodes.EV_KEY:
            key = categorize(event)

            if key.keystate == key.key_down:

                print("KEY:", key.keycode)

                key_str = str(key.keycode)

                # =========================
                # 🎬 SCENE MODE (RESTORED)
                # =========================
                if any(k in key_str for k in ['PLAY', 'PAUSE']):
                    print("SCENE BUTTON DETECTED")

                    if last_frame is not None and not scene_active.is_set():
                        threading.Thread(
                            target=run_scene_description,
                            args=(last_frame.copy(),),
                            daemon=True
                        ).start()

                # =========================
                # ⏭ NEXT SONG → MODE SWITCH (PRIMARY)
                # =========================
                elif 'NEXTSONG' in key_str:
                    print("NEXT → MODE SWITCH")
                    toggle_mode()

                # =========================
                # 🔊 VOLUME UP DOUBLE CLICK (BACKUP)
                # =========================
                elif 'VOLUMEUP' in key_str:

                    now = time.time()

                    if now - last_press_time < DOUBLE_PRESS_WINDOW:
                        print("DOUBLE CLICK → MODE SWITCH")
                        toggle_mode()
                        last_press_time = 0
                    else:
                        last_press_time = now
# =========================
# MAIN
# =========================
if __name__ == '__main__':

    # SENSOR
    shared_sensor_data = mp.Array('i', 64)
    sensor_lock = mp.Lock()

    p = mp.Process(target=run_sensor_process, args=(shared_sensor_data, sensor_lock))
    p.daemon = True
    p.start()

    # YOLO
    print("Loading YOLO...")
    model = YOLO("best.pt")

    # BLIP LOAD (ONE TIME)
    print("Loading BLIP...")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to("cpu")
    blip_model.eval()
    print("BLIP ready")

    # CAMERA
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Camera failed")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    threading.Thread(target=headphone_listener, daemon=True).start()

    last_alert = 0

    print("System running...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            last_frame = frame

            with sensor_lock:
                sensor_data = np.array(shared_sensor_data[:], dtype=np.int32).reshape((8, 8))

            last_sensor = sensor_data

            # 🔥 KEEP CAMERA + SENSOR RUNNING DURING SCENE
            if scene_active.is_set():
                cv2.imshow("Camera", frame)
                cv2.imshow("Sensor", draw_sensor(sensor_data))
                if cv2.waitKey(1) == 27:
                    break
                continue

            results = model(frame, verbose=False)

            with _mode_lock:
                mode_cfg = MODES[current_mode]

            best_distance = None
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

                    cx = int((x1 + x2) / 2 / frame.shape[1] * 8)
                    cy = int((y1 + y2) / 2 / frame.shape[0] * 8)

                    cx = max(0, min(7, cx))
                    cy = max(0, min(7, cy))

                    distances = []
                    for rr in range(max(0, cy-1), min(8, cy+2)):
                        for cc in range(max(0, cx-1), min(8, cx+2)):
                            d = sensor_data[rr, cc]
                            if d > 0:
                                distances.append(d)

                    if len(distances) > 0:
                        dist = min(distances)
                    else:
                        dist = None

                    if cx <= 2:
                        direction = "left"
                    elif cx >= 5:
                        direction = "right"
                    else:
                        direction = "ahead"

                    if best_distance is None or (dist is not None and dist < best_distance):
                        best_distance = dist
                        best_label = label
                        best_direction = direction

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"{label} {dist}mm",
                                (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0,255,0), 2)

            # AUDIO
            if best_label is not None:
                if time.time() - last_alert > 1.5:
            
                    if best_distance is not None and best_distance < 1200:
                        speak_text(f"{best_label} {best_direction}")
                    else:
                        speak_text(f"{best_label} {best_direction}")
            
                    last_alert = time.time()

            # UNKNOWN (Mode 2)
            if mode_cfg["catch_unknown"]:
                center = sensor_data[3:5, 3:5].mean()
                if center > 0 and center < 800:
                    if time.time() - last_alert > 1:
                        speak_text("Unknown obstacle ahead")
                        last_alert = time.time()
            
            with _mode_lock:
                mode_name = MODES[current_mode]["name"]
            
            cv2.putText(frame, f"Mode: {mode_name}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)
                        
            cv2.imshow("Camera", frame)
            cv2.imshow("Sensor", draw_sensor(sensor_data))
            
            if cv2.waitKey(1) == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        p.terminate()
        p.join()
