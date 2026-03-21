import time
import numpy as np


class VL53L5CXSensor:

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.i2c_sensor = None
        self.image_resolution = 0
        self.image_width = 0

        self._init_i2c()

    def _init_i2c(self):
        try:
            import qwiic_vl53l5cx

            self.i2c_sensor = qwiic_vl53l5cx.QwiicVL53L5CX()

            if not self.i2c_sensor.begin():
                raise Exception("Sensor not found. Check wiring or run i2cdetect -y 1")

            self.i2c_sensor.set_resolution(8 * 8)
            self.image_resolution = self.i2c_sensor.get_resolution()
            self.image_width = int(np.sqrt(self.image_resolution))

            self.i2c_sensor.start_ranging()

            if self.verbose:
                print("VL53L5CX ready (Raspberry Pi I2C)")

        except ImportError:
            raise Exception("Run: pip install sparkfun-qwiic-vl53l5cx")

        except Exception as e:
            raise Exception(f"I2C init failed: {e}")

    def is_data_ready(self):
        return self.i2c_sensor.is_data_ready()

    def get_ranging_data(self):
        if not self.i2c_sensor.is_data_ready():
            return None

        try:
            data = self.i2c_sensor.get_ranging_data()

            distances = np.array(data.distance_mm, dtype=np.int32)
            distances = distances.reshape((8, 8))

            return distances

        except Exception as e:
            if self.verbose:
                print(f"Read error: {e}")
            return None

    def read_frame(self):
        return self.get_ranging_data()

    def print_distance_array(self, arr):
        if arr is None:
            return

        for row in arr:
            print("\t".join(str(v) for v in row))
        print()

    def close(self):
        if self.verbose:
            print("Sensor closed")
