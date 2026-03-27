import cv2
import numpy as np
from working_cam_sensor import VL53L5CXSensor

def draw_sensor_grid(sensor_data):
    # Set up a 400x400 pixel window
    size = 400
    cell = size // 8
    img = np.zeros((size, size, 3), dtype=np.uint8)

    for r in range(8):
        for c in range(8):
            d = sensor_data[r, c]

            # Handle colors based on distance
            if d == 0:
                color = (50, 50, 50)  # Dark grey for no reading
            else:
                # Scale distance for color mapping (assuming max range ~3000mm)
                v = min(d, 3000) / 3000.0
                # BGR format: Close = Red, Far = Green
                color = (0, int(255 * v), int(255 * (1 - v)))

            # Draw the colored cell
            x1, y1 = c * cell, r * cell
            x2, y2 = x1 + cell, y1 + cell
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            
            # Overlay the exact distance in millimeters on top of each cell
            cv2.putText(img, str(d), (x1 + 5, y1 + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return img

def main():
    print("Initializing sensor...")
    try:
        # Load your existing sensor class
        sensor = VL53L5CXSensor(verbose=True)
    except Exception as e:
        print(f"Failed to start sensor: {e}")
        return

    print("Sensor running! Press 'ESC' in the window to exit.")

    while True:
        # Check if new distance data is ready
        if sensor.is_data_ready():
            data = sensor.get_ranging_data()
            
            if data is not None:
                # Generate the grid image
                grid_img = draw_sensor_grid(data)
                
                # Show the image in a window
                cv2.imshow("VL53L5CX 8x8 Grid Test", grid_img)
        
        # Required for OpenCV to update the window and listen for the ESC key
        if cv2.waitKey(1) == 27: 
            break

    # Cleanup when finished
    sensor.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
