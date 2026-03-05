import cv2
from ultralytics import YOLO

print("Loading YOLO model...")
model = YOLO("yolov8n.pt")

print("starting camera...")

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("failed to grab frame")
        break
    
    results = model(frame, stream = True)
    
    for result in results:
        annotated_frame = result.plot()
        
        cv2.imshow("live YOLO object detection", annotated_frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("closing...")
        break

cap.release()
cv2.destroyAllWindows()

