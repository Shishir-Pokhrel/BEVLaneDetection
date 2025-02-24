// This is an .ino file. Please save it as such to use this. 

#include <arm_math.h>
#include "Arduino_BHY2.h"
#include "Nicla_System.h"
#include <RGBled.h>

  Sensor device_orientation(SENSOR_ID_DEVICE_ORI);
  SensorOrientation orientation(SENSOR_ID_ORI);  
  
  void setup(){
    Serial.begin(9600); //The serial communication must be at the same rate as the serial communication to python. 
    BHY2.begin();
    orientation.begin();
  }

  void loop(){
    BHY2.update();
    
    Serial.print("orientation pitch :");
    Serial.println(orientation.pitch());

    Serial.print("orientation Heading :");
    Serial.println(orientation.heading());

    Serial.print("orientation roll :");
    Serial.println(orientation.roll());
    delay(50);
  }

