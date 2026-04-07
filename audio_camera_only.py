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
# CPU LIMIT
# =========================
torch.set_num_threads(2)

# =========================
# MODES & CLASSES
# =========================
MODE_NORMAL = 0
MODE_EVERYTHING = 1
MODE_EMERGENCY = 2
current_mode = MODE_NORMAL

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
# SCENE DESCRIPTION
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
        threading.Thread(target=run_scene_description,
                         args=(frame.copy(),),
                         daemon=True).start()

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
        speak_text("Normal mode")
    else:
        current_mode = MODE_EVERYTHING
        speak_text("Everything mode")

def toggle_emergency_mode():
    global current_mode
    if current_mode == MODE_EMERGENCY:
        current_mode = MODE_NORMAL
        speak_text("Normal mode")
    else:
        current_mode = MODE_EMERGENCY
        speak_text("Emergency mode")

# =========================
# HEADPHONE LISTENER
# =========================
def headphone_listener():
    paths = list_devices()
    sel = selectors.DefaultSelector()

    for path in paths:
        try:
            dev = InputDevice(path)
            sel.register(dev, selectors.EVENT_READ)
        except PermissionError:
            pass

    while True:
        for key, _ in sel.select():
            dev = key.fileobj
            for event in dev.read():
                if event.type == ecodes.EV_KEY:
                    k = categorize(event)

                    if k.keystate == k.key_down:
                        if k.keycode in ['KEY_PLAYCD', 'KEY_PLAYPAUSE']:
                            trigger_scene_global()
                        elif k.keycode == 'KEY_NEXTSONG':
                            toggle_everything_mode()
                        elif k.keycode == 'KEY_PREVIOUSSONG':
                            toggle_emergency_mode()

# =========================
# MAIN
# =========================
if __name__ == '__main__':

    model = YOLO("best6_ncnn_model")

    from transformers import BlipProcessor, BlipForConditionalGeneration
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to("cpu")
    blip_model.eval()

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    threading.Thread(target=headphone_listener, daemon=True).start()

    last_alert = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            latest_frame = frame

            # MODE LOGIC
            if current_mode == MODE_NORMAL:
                active_classes = OBSTACLE_CLASSES - {"Person"}
                catch_unknown = False
            elif current_mode == MODE_EVERYTHING:
                active_classes = OBSTACLE_CLASSES
                catch_unknown = True
            else:
                active_classes = OBSTACLE_CLASSES
                catch_unknown = False

            if not scene_active.is_set():
                results = model(frame, verbose=False)
                yolo_alert_triggered = False

                for r in results:
                    for b in r.boxes:
                        cls = int(b.cls[0])
                        label = model.names[cls]

                        if label not in active_classes:
                            continue

                        x1, y1, x2, y2 = map(int, b.xyxy[0])

                        # ===== DIRECTION ONLY =====
                        cx_norm = (x1 + x2) / 2 / frame.shape[1]

                        if cx_norm < 0.33:
                            direction = "left"
                        elif cx_norm < 0.66:
                            direction = "center"
                        else:
                            direction = "right"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} ({direction})",
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 2)

                        # ONLY ONE ALERT PER FRAME + COOLDOWN
                        if not yolo_alert_triggered:
                                speak_text(f"{label} on your {direction}")
                                last_alert = time.time()
                            yolo_alert_triggered = True

                # UNKNOWN fallback (no distance)
                if catch_unknown and not yolo_alert_triggered:
                    if time.time() - last_alert > 2:
                        speak_text("Unknown object ahead")
                        last_alert = time.time()

            cv2.imshow("Camera", frame)

            if cv2.waitKey(1) == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
