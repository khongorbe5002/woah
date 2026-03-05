import cv2
from ultralytics import YOLO

print ("loading yolo model")
model = YOLO("yolov8n.pt")

print("running inference on image")
results = model("sample.jpg")

annotated_image = results[0].plot()

print("displaying results, press any key to close")
cv2.imshow("yolo object detection", annotated_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

