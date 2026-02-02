"""
VL53L5CX ToF Sensor Interface for Python
Converts Arduino code to Python equivalent
Supports both direct I2C (via SparkFun library) and serial communication
"""

import serial
import serial.tools.list_ports
import time
import numpy as np

class VL53L5CXSensor:
    """Python interface for VL53L5CX sensor - equivalent to Arduino code"""
    
    def __init__(self, port=None, baudrate=921600, use_serial=True):
        """
        Initialize sensor
        
        Args:
            port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
                 If None, will auto-detect ESP32
            baudrate: Serial baud rate (default 921600 to match ESP32)
            use_serial: If True, use serial communication. If False, use direct I2C
        """
        self.use_serial = use_serial
        self.image_resolution = 0
        self.image_width = 0
        self.serial_conn = None
        self.i2c_sensor = None
        
        if use_serial:
            self._init_serial(port, baudrate)
        else:
            self._init_i2c()
    
    def _init_serial(self, port, baudrate):
        """Initialize serial communication with ESP32"""
        if port is None:
            # Auto-detect ESP32 (looks for common ESP32 USB serial chips)
            ports = serial.tools.list_ports.comports()
            for p in ports:
                # Common ESP32 identifiers
                if 'CH340' in p.description or 'CP210' in p.description or 'USB Serial' in p.description:
                    port = p.device
                    print(f"Auto-detected ESP32 on {port}")
                    break
            
            if port is None and len(ports) > 0:
                # Use first available port as fallback
                port = ports[0].device
                print(f"Using first available port: {port}")
        
        try:
            self.serial_conn = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Wait for ESP32 to initialize
            print(f"Connected to ESP32 on {port}")
        except Exception as e:
            print(f"Error connecting to serial port: {e}")
            print("Make sure ESP32 is connected and upload the sensor code to it")
            raise
    
    def _init_i2c(self):
        """Initialize direct I2C connection (requires sparkfun-qwiic-vl53l5cx)"""
        try:
            import qwiic_vl53l5cx  # type: ignore
            self.i2c_sensor = qwiic_vl53l5cx.QwiicVL53L5CX()
            
            if not self.i2c_sensor.begin():
                raise Exception("Sensor not found - check your wiring")
            
            self.i2c_sensor.set_resolution(8*8)  # Enable all 64 pads
            self.image_resolution = self.i2c_sensor.get_resolution()
            self.image_width = int(np.sqrt(self.image_resolution))
            self.i2c_sensor.start_ranging()
            print("VL53L5CX sensor initialized via I2C")
        except ImportError:
            print("I2C mode requires: pip install sparkfun-qwiic-vl53l5cx")
            raise
        except Exception as e:
            print(f"Error initializing I2C sensor: {e}")
            raise
    
    def is_data_ready(self):
        """Check if sensor data is ready"""
        if self.use_serial:
            # For serial mode, we'll read when data is available
            return self.serial_conn.in_waiting > 0
        else:
            return self.i2c_sensor.is_data_ready()
    
    def get_ranging_data(self):
        """
        Read distance data into array (equivalent to Arduino getRangingData)
        Returns: 8x8 numpy array of distances in mm, or None if no data
        """
        if self.use_serial:
            return self._read_serial_data()
        else:
            return self._read_i2c_data()
    
    def _read_serial_data(self):
        """Read sensor data from ESP32 via serial"""
        if not self.serial_conn.in_waiting:
            return None
        
        # Read all available data
        data = self.serial_conn.read(self.serial_conn.in_waiting).decode('utf-8', errors='ignore')
        
        # Debug: print raw data
        print(f"Raw data received: {repr(data[:200])}")
        
        # Parse the 8x8 array from serial output
        # Format: 8 rows of tab or space-separated values, separated by newline
        # Each block is separated by a space line
        lines = data.split('\n')
        distances = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and space separators
            if not line or line == " ":
                # If we have 8 rows collected, return them
                if len(distances) == 8:
                    return np.array(distances)
                continue
            
            # Try to parse row (tab or space separated)
            try:
                # Split by tab or space
                values = []
                for val in line.split():
                    values.append(int(val))
                
                if len(values) == 8:
                    distances.append(values)
                    # Return immediately when we get 8 rows
                    if len(distances) == 8:
                        return np.array(distances)
            except ValueError:
                # If we can't parse a line as numbers, reset
                distances = []
                continue
        
        return None
    
    def _read_i2c_data(self):
        """Read sensor data directly via I2C"""
        if not self.i2c_sensor.is_data_ready():
            return None
        
        measurement_data = self.i2c_sensor.get_ranging_data()
        
        # Convert to 8x8 numpy array
        # The ST library returns data transposed, so we need to reshape
        distances = np.array(measurement_data.distance_mm)
        distances_2d = distances.reshape((8, 8))
        
        # Transpose to match Arduino output format (increasing y, decreasing x)
        return distances_2d
    
    def print_distance_array(self, distance_array):
        """
        Pretty-print distance array (equivalent to Arduino print format)
        The ST library returns the data transposed from zone mapping shown in datasheet
        Pretty-print data with increasing y, decreasing x to reflect reality
        """
        if distance_array is None:
            return
        
        image_width = distance_array.shape[1]
        
        # Print with increasing y, decreasing x (like Arduino code)
        for y in range(image_width):
            row = []
            for x in range(image_width - 1, -1, -1):
                row.append(f"{distance_array[y, x]}")
            print("\t".join(row))
        print()
        print()
    
    def close(self):
        """Close connections"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        print("Sensor connection closed")
