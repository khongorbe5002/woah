import time
from vl53l5cx_sensor import VL53L5CXSensor

sensor = VL53L5CXSensor(use_serial=False)

print("Sensor running...")

while True:
    data = sensor.get_ranging_data()

    if data is not None:
        sensor.print_distance_array(data)

    time.sleep(0.05)
