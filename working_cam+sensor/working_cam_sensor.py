import time
import numpy as np


class VL53L5CXSensor:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.i2c_sensor = None
        self._init_i2c()

    def _init_i2c(self):
        try:
            import qwiic_vl53l5cx

            self.i2c_sensor = qwiic_vl53l5cx.QwiicVL53L5CX()

            if not self.i2c_sensor.begin():
                raise Exception("Sensor not found. Check wiring (run i2cdetect -y 1)")

            self.i2c_sensor.set_resolution(8 * 8)
            self.i2c_sensor.start_ranging()

            if self.verbose:
                print("VL53L5CX initialized (I2C)")

        except ImportError:
            raise Exception("Install: pip install sparkfun-qwiic-vl53l5cx")

    def get_ranging_data(self):
        if not self.i2c_sensor.is_data_ready():
            return None

        try:
            data = self.i2c_sensor.get_ranging_data()
            arr = np.array(data.distance_mm, dtype=np.int32).reshape((8, 8))
            return arr
        except:
            return None

    def close(self):
        if self.verbose:
            print("Sensor closed")

# MAIN LOOP
sensor = VL53L5CXSensor(verbose=True)

print("\nSensor running...\n")

try:
    while True:
        data = sensor.get_ranging_data()

        if data is not None:
            # clear terminal
            print("\033[H\033[J", end="")

            print("Distance Grid (mm):\n")

            for row in data:
                print("\t".join(f"{v if v > 0 else '---':>4}" for v in row))

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    sensor.close()
