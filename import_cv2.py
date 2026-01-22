from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov10n.pt")
cap = cv2.VideoCapture(0)

OBSTACLE_CLASSES = [
    "person", "bicycle", "car", "chair", "bench", "truck", "traffic light",
    "fire hydrant", "stop sign", "stairs","phone","scooter","motorcycle" # stairs requires custom model later
]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, stream=True)
    
    for r in results:
        boxes = r.boxes
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0]
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            
            # Draw box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

            label = model.names[cls]

            if label in OBSTACLE_CLASSES and conf > 0.6:
                cv2.putText(frame, f"This is a:{label} with:{conf:.1f} conf", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                print(f"ALERT: Object detected, this is a:{label} with:{conf:.1f} conf")
                print(f"  Location: Top-left ({int(x1)}, {int(y1)}) | Bottom-right ({int(x2)}, {int(y2)}) | Center ({int((x1+x2)/2)}, {int((y1+y2)/2)})")
            
    cv2.imshow("Obstacle Detection Demo", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()




#if label in OBSTACLE_CLASSES and conf > 0.5:
    
