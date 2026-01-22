from ultralytics import YOLO
import cv2
import time
import numpy as np

model = YOLO("yolov10n.pt")
cap = cv2.VideoCapture(0)

OBSTACLE_CLASSES = [
    "person", "bicycle", "car", "chair", "bench", "truck", "traffic light",
    "fire hydrant", "stop sign", "stairs","phone","scooter","motorcycle" # stairs requires custom model later
]

# Grid visualization parameters
GRID_WIDTH = 800
GRID_HEIGHT = 600
GRID_CELL_SIZE = 50  # Size of each grid square
GRID_COLOR = (50, 50, 50)  # Dark gray for grid lines
TEXT_COLOR = (0, 255, 0)  # Green text
BOX_COLOR = (0, 255, 0)  # Green boxes

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

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Create grid canvas for this frame
    grid_canvas = create_grid_canvas()
    
    results = model(frame, stream=True)
    
    for r in results:
        boxes = r.boxes
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0]
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            
            # Draw box on main frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

            label = model.names[cls]

            if label in OBSTACLE_CLASSES and conf > 0.6:
                cv2.putText(frame, f"This is a:{label} with:{conf:.1f} conf", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
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
            
    cv2.imshow("Obstacle Detection Demo", frame)
    cv2.imshow("Filtered Detections - Grid View", grid_canvas)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()




#if label in OBSTACLE_CLASSES and conf > 0.5:
    