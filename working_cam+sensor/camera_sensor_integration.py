from ultralytics import YOLO
import cv2
import time
import numpy as np
from vl53l5cx_sensor import VL53L5CXSensor

model = YOLO("yolov10n.pt")
cap = cv2.VideoCapture(0)

# Initialize VL53L5CX sensor
# For serial communication (ESP32), use: sensor = VL53L5CXSensor(port='COM3')  # Replace with your COM port
# For direct I2C, use: sensor = VL53L5CXSensor(use_serial=False)
# Auto-detect ESP32 on serial port:
try:
    sensor = VL53L5CXSensor(port=None, use_serial=True)  # Auto-detect ESP32
    print("VL53L5CX sensor initialized")
except Exception as e:
    print(f"Warning: Could not initialize sensor: {e}")
    print("Continuing without sensor data...")
    sensor = None

OBSTACLE_CLASSES = [
    "person", "bicycle", "car", "chair", "bench", "truck", "traffic light",
    "fire hydrant", "stop sign", "stairs","phone","scooter","motorcycle",
    "tree","bushes","cup","cups","bowl"  # stairs requires custom model later
]

# Grid visualization parameters
GRID_WIDTH = 800
GRID_HEIGHT = 600
GRID_CELL_SIZE = 50  # Size of each grid square
GRID_COLOR = (50, 50, 50)  # Dark gray for grid lines
TEXT_COLOR = (0, 255, 0)  # Green text
BOX_COLOR = (0, 255, 0)  # Green boxes

# Sensor grid visualization parameters
SENSOR_GRID_SIZE = 600  # Size of sensor grid window (square)
SENSOR_CELL_SIZE = SENSOR_GRID_SIZE // 8  # Each cell is 1/8 of the grid
SENSOR_MAX_DISTANCE = 3300  # Maximum distance in mm for color scaling (raw sensor max)

def create_grid_canvas():
    """Create a black canvas with a grid"""
    canvas = np.zeros((GRID_HEIGHT, GRID_WIDTH, 3), dtype=np.uint8)
    
    # Draw vertical lines
    for x in range(0, GRID_WIDTH, GRID_CELL_SIZE):
        cv2.line(canvas, (x, 0), (x, GRID_HEIGHT), GRID_COLOR, 1)
    
    # Draw horizontal lines
    for y in range(0, GRID_HEIGHT, GRID_CELL_SIZE):
        cv2.line(canvas, (0, y), (GRID_WIDTH, y), GRID_COLOR, 1)
    
    return canvas

def create_sensor_grid(sensor_data):
    """
    Create a visualization grid for the 8x8 TOF sensor data
    Color-codes cells based on distance (closer = red/orange, farther = blue/green)
    """
    canvas = np.zeros((SENSOR_GRID_SIZE, SENSOR_GRID_SIZE, 3), dtype=np.uint8)
    
    # Draw grid lines
    for i in range(9):
        x = i * SENSOR_CELL_SIZE
        cv2.line(canvas, (x, 0), (x, SENSOR_GRID_SIZE), GRID_COLOR, 2)
        cv2.line(canvas, (0, x), (SENSOR_GRID_SIZE, x), GRID_COLOR, 2)
    
    # Fill cells with color based on distance
    for row in range(8):
        for col in range(8):
            distance = sensor_data[row, col]
            
            # Calculate cell position
            x1 = col * SENSOR_CELL_SIZE
            y1 = row * SENSOR_CELL_SIZE
            x2 = (col + 1) * SENSOR_CELL_SIZE
            y2 = (row + 1) * SENSOR_CELL_SIZE
            
            # Color mapping: closer objects = red/orange, farther = blue/green
            # Use raw distance in mm (0-3300), clamp to max distance
            clamped_dist = min(distance, SENSOR_MAX_DISTANCE)
            
            if distance == 0:
                # Invalid - gray
                color = (50, 50, 50)
            else:
                # Color gradient: red (close) -> yellow -> green -> blue (far)
                # Map 0-3300 to 0-1
                normalized_dist = clamped_dist / SENSOR_MAX_DISTANCE
                
                if normalized_dist < 0.25:
                    # Close: Red to Orange
                    r = 255
                    g = int(255 * (normalized_dist / 0.25))
                    b = 0
                elif normalized_dist < 0.5:
                    # Medium-close: Orange to Yellow
                    r = 255
                    g = 255
                    b = int(255 * ((normalized_dist - 0.25) / 0.25))
                elif normalized_dist < 0.75:
                    # Medium-far: Yellow to Green
                    r = int(255 * (1 - (normalized_dist - 0.5) / 0.25))
                    g = 255
                    b = 0
                else:
                    # Far: Green to Blue
                    r = 0
                    g = int(255 * (1 - (normalized_dist - 0.75) / 0.25))
                    b = int(255 * ((normalized_dist - 0.75) / 0.25))
                
                color = (int(b), int(g), int(r))  # BGR format for OpenCV
            
            # Fill cell with color
            cv2.rectangle(canvas, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), color, -1)
            
            # Add distance text (in mm, or '---' if invalid)
            if distance == 0 or distance > SENSOR_MAX_DISTANCE:
                text = "---"
            else:
                text = f"{int(distance)}"
            
            # Calculate text position (centered in cell)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            text_x = x1 + (SENSOR_CELL_SIZE - text_size[0]) // 2
            text_y = y1 + (SENSOR_CELL_SIZE + text_size[1]) // 2
            
            # Use white or black text based on cell brightness
            brightness = sum(color) / 3
            text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)
            
            cv2.putText(canvas, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    
    # Add title
    cv2.putText(canvas, "TOF Sensor Grid (mm)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return canvas

# Store last sensor data to retain visualization
last_sensor_data = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip camera horizontally
    frame = cv2.flip(frame, 1)
    
    # Create grid canvas for this frame
    grid_canvas = create_grid_canvas()
    
    # Read TOF sensor data - update only if new data is available
    if sensor and sensor.is_data_ready():
        new_data = sensor.get_ranging_data()
        if new_data is not None:
            last_sensor_data = new_data
    
    # Use last sensor data for visualization (retains previous data if no new data)
    sensor_data = last_sensor_data
    
    results = model(frame, stream=True)
    
    for r in results:
        boxes = r.boxes
        for b in boxes:
            #bounding box locations USE FOR SENSOR
            x1, y1, x2, y2 = b.xyxy[0]
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            
            #draws box on main frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

            label = model.names[cls]

            if label in OBSTACLE_CLASSES and conf > 0.4:
                cv2.putText(frame, f"This is a:{label} with:{conf:.1f} conf", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 2)
                print(f"ALERT: Object detected, this is a:{label} with:{conf:.1f} conf")
                print(f"  Location: Top-left ({int(x1)}, {int(y1)}) | Bottom-right ({int(x2)}, {int(y2)}) | Center ({int((x1+x2)/2)}, {int((y1+y2)/2)})")
                
                # Scale and draw on grid canvas
                # Normalize coordinates to grid size
                grid_x1 = int((x1 / frame.shape[1]) * GRID_WIDTH)
                grid_y1 = int((y1 / frame.shape[0]) * GRID_HEIGHT)
                grid_x2 = int((x2 / frame.shape[1]) * GRID_WIDTH)
                grid_y2 = int((y2 / frame.shape[0]) * GRID_HEIGHT)
                
                # Clamp to grid bounds
                grid_x1 = max(0, min(grid_x1, GRID_WIDTH - 1))
                grid_y1 = max(0, min(grid_y1, GRID_HEIGHT - 1))
                grid_x2 = max(0, min(grid_x2, GRID_WIDTH - 1))
                grid_y2 = max(0, min(grid_y2, GRID_HEIGHT - 1))
                
                # Draw box on grid
                cv2.rectangle(grid_canvas, (grid_x1, grid_y1), (grid_x2, grid_y2), BOX_COLOR, 2)
                
                # Add label text on grid
                cv2.putText(grid_canvas, f"{label} ({conf:.1f})", (grid_x1, grid_y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    
    # Create sensor visualization grid (use last data, or empty grid if no data yet)
    if sensor_data is not None:
        sensor_grid = create_sensor_grid(sensor_data)
    else:
        # Create empty grid until first data arrives
        sensor_grid = np.zeros((SENSOR_GRID_SIZE, SENSOR_GRID_SIZE, 3), dtype=np.uint8)
        cv2.putText(sensor_grid, "Waiting for sensor data...", (SENSOR_GRID_SIZE//2 - 150, SENSOR_GRID_SIZE//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
    
    cv2.imshow("Obstacle Detection Demo", frame)
    cv2.imshow("Filtered Detections - Grid View", grid_canvas)
    cv2.imshow("TOF Sensor Grid", sensor_grid)
    
    if cv2.waitKey(1) == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
if sensor:
    sensor.close()




#if label in OBSTACLE_CLASSES and conf > 0.5:
    
