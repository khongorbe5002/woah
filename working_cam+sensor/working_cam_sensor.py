import time
import numpy as np
import cv2


class VL53L5CXSensor:
    def __init__(self, use_serial=False, verbose=True):
        self.use_serial = use_serial
        self.verbose = verbose
        self.image_resolution = 0
        self.image_width = 0
        self.i2c_sensor = None

        if self.use_serial:
            raise Exception("Serial mode disabled. Use use_serial=False for Raspberry Pi.")
        else:
            self._init_i2c()

    def _init_i2c(self):
        try:
            import qwiic_vl53l5cx

            self.i2c_sensor = qwiic_vl53l5cx.QwiicVL53L5CX()

            if not self.i2c_sensor.begin():
                raise Exception("Sensor not found. Run: i2cdetect -y 1")

            self.i2c_sensor.set_resolution(8 * 8)
            self.image_resolution = self.i2c_sensor.get_resolution()
            self.image_width = int(np.sqrt(self.image_resolution))

            self.i2c_sensor.start_ranging()

            if self.verbose:
                print("VL53L5CX initialized (I2C mode)")

        except ImportError:
            raise Exception("pip install sparkfun-qwiic-vl53l5cx")

    def get_ranging_data(self):
        if not self.i2c_sensor.is_data_ready():
            return None

        try:
            data = self.i2c_sensor.get_ranging_data()
            distances = np.array(data.distance_mm, dtype=np.int32).reshape((8, 8))
            return distances
        except:
            return None

    def close(self):
        if self.verbose:
            print("Sensor closed")


# ─────────────────────────────────────
# MAIN
# ─────────────────────────────────────

sensor = VL53L5CXSensor(use_serial=False, verbose=False)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("Camera failed")
    cap = None

print("Running...\n")

try:
    while True:
        # --- Camera ---
        if cap:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed")

        # --- Sensor ---
        data = sensor.get_ranging_data()

        if data is not None:
            print("\033[H\033[J", end="")  # clear terminal

            print("Distance Grid (mm):\n")
            for row in data:
                print("\t".join(f"{v if v > 0 else '---':>4}" for v in row))

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    if cap:
        cap.release()
    sensor.close()
