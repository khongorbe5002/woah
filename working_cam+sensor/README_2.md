notes:

- must upload .ino file to the ESP-32 before running program
- laptops have a value associated to each "camera" ( camera (0), camera(1), etc.). when you connect the external camera this number may not match the one in the main .py program, so change the cv.VideoCapture(x) if you initially see your personal camera in the pop up
- the secondary .py program runs in the background and connects the sensor to the main program
