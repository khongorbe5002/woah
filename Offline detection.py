from ultralytics import YOLO
import os

# 1. Load your trained YOLOv10 weights
# Using nano model for maximum speed: 'yolov10n.pt'
model = YOLO('yolov10n.pt') 

# 2. Run inference on your pre-recorded video
video_source = 'IKB.mp4'
if not os.path.exists(video_source):
    print(f"Error: Video file '{video_source}' not found.")
    exit(1)
print(f"Processing video: {video_source}")
output_name = video_source.replace('.mp4', '').replace(' ', '_')
print("Starting inference... Press 'q' in the video window to quit, or 'space' to pause/resume.")
print("You can also press Ctrl+C in the terminal to interrupt.")
try:
    results = model.predict(
        source=video_source,
        project='runs/detect',
        name=output_name,  # Unique folder for each video
        save=True,          # This tells YOLO to save a new video with the bounding boxes drawn
        conf=0.25,          # Lower confidence for faster processing (fewer false positives)
        vid_stride=3,       # Process every 2nd frame for 2x speed (may reduce accuracy slightly)
        imgsz=416,          # Smaller image size for faster inference (default 640)
        show=True           # Optional: Opens a window to watch the processing happen in real-time
    )
    print(f"Offline processing complete. Check the 'runs/detect/{output_name}' folder for the output video.")
except KeyboardInterrupt:
    print("\nProcessing interrupted by user. Check the 'runs/detect/{output_name}' folder for any partial output.")