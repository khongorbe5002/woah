/*
  ESP32 Code for VL53L5CX Sensor - Python Integration
  Modified from SparkFun example to send structured data to Python
  
  This code reads the 8x8 distance array and sends it to Python via Serial
  in a format that's easy to parse.
  
  Upload this to your ESP32, then run the Python camera script.
*/

#include <Wire.h>
#include <SparkFun_VL53L5CX_Library.h>

SparkFun_VL53L5CX myImager;
VL53L5CX_ResultsData measurementData; // Result data class structure, 1356 bytes of RAM

int imageResolution = 0;
int imageWidth = 0;

void setup()
{
  Serial.begin(115200);
  delay(1000);
  Serial.println("ESP32 VL53L5CX Sensor - Python Integration");
  Serial.println("Ready to send data...");

  Wire.begin(); // This resets to 100kHz I2C
  Wire.setClock(400000); // Sensor has max I2C freq of 400kHz 
  
  Serial.println("Initializing sensor board. This can take up to 10s. Please wait.");
  if (myImager.begin() == false)
  {
    Serial.println(F("Sensor not found - check your wiring. Freezing"));
    while (1) ;
  }
  
  myImager.setResolution(8*8); // Enable all 64 pads
  
  imageResolution = myImager.getResolution(); // Query sensor for current resolution - either 4x4 or 8x8
  imageWidth = sqrt(imageResolution); // Calculate printing width

  myImager.startRanging();
  Serial.println("Sensor initialized. Starting data transmission...");
}

void loop()
{
  // Poll sensor for new data
  if (myImager.isDataReady() == true)
  {
    if (myImager.getRangingData(&measurementData)) // Read distance data into array
    {
      // Send data in structured format for Python parsing
      // Format: Each row on a new line, tab-separated values
      // The ST library returns the data transposed from zone mapping shown in datasheet
      // Send with increasing y, decreasing x to reflect reality (like original Arduino code)
      for (int y = 0 ; y <= imageWidth * (imageWidth - 1) ; y += imageWidth)
      {
        for (int x = imageWidth - 1 ; x >= 0 ; x--)
        {
          Serial.print(measurementData.distance_mm[x + y]);
          if (x > 0) Serial.print("\t");
        }
        Serial.println();
      }
      // Send delimiter to indicate end of data block
      Serial.println("END_DATA");
    }
  }

  delay(5); // Small delay between polling
}
