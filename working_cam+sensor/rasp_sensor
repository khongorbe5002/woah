import time
import numpy as np
import matplotlib.pyplot as plt
from vl53l5cx import VL53L5CX

sensor = VL53L5CX()

print("Initializing sensor...")
sensor.set_resolution(8*8)
sensor.set_ranging_frequency_hz(15)
sensor.start_ranging()

plt.ion()

while True:
    if sensor.data_ready():
        data = sensor.get_data()

        # distance_mm is a flat list of 64 values
        distances = np.array(data.distance_mm).reshape((8, 8))

        # Flip to match real-world orientation (like your ESP32 code)
        distances = np.fliplr(distances)

        print("\nFrame:")
        print(distances)

        plt.clf()
        plt.imshow(distances, cmap='viridis')
        plt.colorbar(label="Distance (mm)")
        plt.pause(0.01)

    time.sleep(0.01)
