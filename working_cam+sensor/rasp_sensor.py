import time
import numpy as np
from vl53l5cx_sensor import VL53L5CXSensor

sensor = VL53L5CXSensor()

print("Initializing sensor...")
sensor.start_ranging()

while True:
    if sensor.check_data_ready():
        data = sensor.get_ranging_data()

        # Convert to 8x8 grid
        distances = np.array(data).reshape((8, 8))

        print("\nFrame:")
        print(distances)

    time.sleep(0.05)
