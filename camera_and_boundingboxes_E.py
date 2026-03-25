from ultralytics import YOLO
import cv2
import time
import threading
import numpy as np
import subprocess

from working_cam_sensor import VL53L5CXSensor

def speak_text(text):
    try:
        subprocess.Popen(["espeak", text])
    except:
        pass


model = YOLO("best_ncnn_model")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("Camera failed")
    exit()


try:
    sensor = VL53L5CXSensor(verbose=False)
except Exception as e:
    print(f"Sensor failed: {e}")
    sensor = None


last_sensor = np.zeros((8, 8), dtype=np.int32)
sensor_lock = threading.Lock()


def sensor_thread():
    global last_sensor
    while True:
        if sensor is None:
            time.sleep(0.1)
            continue

        data = sensor.get_ranging_data()
        if data is not None:
            with sensor_lock:
                last_sensor = np.array(data)

        time.sleep(0.01)


threading.Thread(target=sensor_thread, daemon=True).start()


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


last_alert = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    with sensor_lock:
        sensor_data = last_sensor.copy()

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

    grid = draw_sensor(sensor_data)

    cv2.imshow("Camera", frame)
    cv2.imshow("Sensor", grid)

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()

if sensor:
    sensor.close()
