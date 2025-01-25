# EEN210

# To run the code
1. Run the following lines to download required packages:
    - python -m venv .venv  # "py -3.11 -m venv .venv" #python >3.12 does not work
    - for ***Mac*** and ***Linux***:  
        - source .venv/bin/activate  
    - for ***Windows***:  
        - run terminal(vscode) with administrative privilage  
        - Set-ExecutionPolicy RemoteSigned  
        - .venv/Scripts/activate  # ".\.venv\Scripts\activate" if using PowerShell
    - pip install -r requirements.txt  
2. You need to change ***"Your_IP_Address"*** in index.html file;  
3. You need to also add ***"your WiFi SSID"***, ***"your Passoword"*** and ***"your WiFi IP Address"*** in main.cpp if you want to programm the micro controler.  


# REGARDING platform.ini   .... use your own local file!!!
# Consider the following configuration or your own:

; PlatformIO Project Configuration File
;
;   Build options: build flags, source filter
;   Upload options: custom upload port, speed and extra flags
;   Library options: dependencies, extra library storages
;   Advanced options: extra scripting
;
; Please visit documentation for the other options and examples
; https://docs.platformio.org/page/projectconf.html'
;    adafruit/Adafruit MPU6050 @ ^2.0.3

;   links2004/WebSockets@^2.4.1
;  links2004/WebSockets@^2.3.6
;    links2004/WebSockets@^2.4.1
;    WebSocketClient
;    ArduinoWebsockets

[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
lib_deps =
    adafruit/Adafruit Unified Sensor @ ^1.1.4
    MPU6050
    adafruit/Adafruit MPU6050 @ ^2.0.3
    PubSubClient
    AsyncMqttClient
    AsyncTCP
    ArduinoJson
    HTTPClient
    WebSocketsClient
    WiFiMulti
    Wire
    ArduinoWebsockets
    WebSockets
    adafruit/Adafruit ADXL345@^1.3.4
    adafruit/Adafruit BusIO@^1.14.5