import cv2
import time
import math
# Remove the pyserial import since you aren't using the ESP32 anymore
# import serial 

# Add your specific Raspberry Pi GPIO/I2C libraries here
import board
import busio
# import adafruit_vl53l5cx # Example: If you use a VL53L5CX 8x8 Time-of-Flight sensor
from ultralytics import YOLO

# --- 1. Initialize Sensor (Replaces ESP32 Serial Connection) ---
# Set up the I2C bus and the sensor
i2c = busio.I2C(board.SCL, board.SDA)
# sensor = adafruit_vl53l5cx.VL53L5CX(i2c)
# sensor.start_ranging()
print("Sensor initialized via I2C.")

# --- 2. Initialize Camera ---
# Use the correct camera index for the Pi (usually 0 for standard USB webcams)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- 3. Initialize YOLO Model ---
# Load your exported model (using .ncnn or .tflite is highly recommended for the Pi!)
# model = YOLO("best_ncnn_model") 
model = YOLO("yolov10n.pt") # Kept as .pt for placeholder, but optimize this later

# Add your class names (assuming standard COCO or your custom classes)
classNames = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train"] # ... etc

# --- 4. Main Processing Loop ---
try:
    while True:
        # A. Read from Sensor (Directly from I2C, no serial string parsing needed!)
        # if sensor.data_ready:
        #     distance_array = sensor.distance
        #     # The array usually comes in as a flat list of 64 values, or an 8x8 matrix
        
        # B. Read from Camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # C. Run YOLO Inference
        # stream=True is a generator and is much faster for video feeds
        results = model(frame, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw box on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Get confidence and class
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]

                # D. Obstacle Logic (Combine Vision + Sensor)
                # This is where your capstone logic shines. Example: If a hazard is detected 
                # in the upper half of the frame AND the sensor reads a close distance, trigger alert.
                # if class_name in ['person', 'sign'] and min(distance_array) < 100:
                #     trigger_haptic_feedback()

        # Display the frame 
        # Note: Disable this imshow line if running "headless" on a wearable Pi to save CPU
        cv2.imshow("Wearable Camera Feed", frame)
        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("Program stopped manually.")

finally:
    # Cleanup resources safely
    # sensor.stop_ranging()
    cap.release()
    cv2.destroyAllWindows()
