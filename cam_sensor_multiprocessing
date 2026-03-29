import cv2
import time
import numpy as np
import subprocess
import multiprocessing as mp
from ultralytics import YOLO

def speak_text(text):
    try:
        # Using espeak for the visually impaired user audio feedback
        subprocess.Popen(["espeak", text])
    except:
        pass

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

# --- 1. THE ISOLATED SENSOR PROCESS ---
# This runs on a completely separate CPU core.
# We MUST initialize the sensor inside this function so the I2C connection belongs to this core.
def run_sensor_process(shared_array, lock):
    from working_cam_sensor import VL53L5CXSensor
    
    try:
        sensor = VL53L5CXSensor(verbose=False)
    except Exception as e:
        print(f"Sensor failed to initialize in background process: {e}")
        return

    while True:
        data = sensor.get_ranging_data()
        if data is not None:
            # The 8x8 matrix must be flattened to a 1D list of 64 items to pass through the memory bridge
            flat_data = data.flatten()
            
            with lock:
                for i in range(64):
                    shared_array[i] = flat_data[i]
        
        # Polling at ~30Hz. This will no longer be interrupted by YOLO!
        time.sleep(0.033) 

# --- 2. THE MAIN EXECUTION BLOCK ---
# Multiprocessing requires this if __name__ == '__main__' guard to prevent infinite spawn loops
if __name__ == '__main__':
    
    # Create the bridge: An array of 64 integers ('i') and a lock to prevent data corruption
    shared_sensor_data = mp.Array('i', 64)
    sensor_lock = mp.Lock()

    # Start the background process
    print("Starting sensor on separate CPU core...")
    p = mp.Process(target=run_sensor_process, args=(shared_sensor_data, sensor_lock))
    p.daemon = True # Ensures the process dies when you close the main script
    p.start()

    # Initialize YOLO and Camera on the main core
    print("Loading YOLO model...")
    model = YOLO("best.pt") # Remember to swap this to NCNN later for even more speed!
    
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Camera failed")
        exit()

    last_alert = 0

    print("System running. Press ESC to exit.")
    
    # --- 3. THE MAIN VISION LOOP ---
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Safely grab the latest 64 integers from the other CPU core
            with sensor_lock:
                current_shared_data = shared_sensor_data[:]
            
            # Rebuild it back into an 8x8 numpy array for your distance logic
            sensor_data = np.array(current_shared_data, dtype=np.int32).reshape((8, 8))

            # Run YOLO (This will stall the main core, but the sensor core will keep running!)
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

    except KeyboardInterrupt:
        print("\nManual exit detected.")
        
    finally:
        # Clean up resources
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        # Kill the background process securely
        p.terminate()
        p.join()
