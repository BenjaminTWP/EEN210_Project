#include <Wire.h>
#include <MPU6050.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <Adafruit_BusIO_Register.h>
#include "credentials.h"

// Public internet, allow port in firewall
// Replace with your network credentials
//const char *ssid = "Name of internet connection goes here";
//const char *password = "Password goes here";

// Replace with your WebSocket server address
//const char *webSocketServer = "ipv-adress goes here";

const int webSocketPort = 8000;
const char *webSocketPath = "/";
MPU6050 mpu; // Define the sensor
WebSocketsClient client;
// SocketIOClient socketIO;

bool wifiConnected = false;

void setup()
{
  Serial.begin(9600);
  Wire.begin();
  mpu.initialize();

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.print("Connected to WiFi network with IP Address: ");
  Serial.println(WiFi.localIP());
  client.begin(webSocketServer, webSocketPort, "/ws");
  Serial.println(client.isConnected());
  wifiConnected = true;
}

void loop()
{
  if (wifiConnected)
  {
    client.loop();

    // Get sensor data
    int16_t ax, ay, az;
    int16_t gx, gy, gz;
    mpu.getAcceleration(&ax, &ay, &az);
    mpu.getRotation(&gx, &gy, &gz);

    // Normalize sensor data to units of g
    double Dax = 9.82 * ax/16384;
    double Day = 9.82 * ay/16384;
    double Daz = 9.82 * az/16384;
    
    double Dgx = gx/131;
    double Dgy = gy/131;
    double Dgz = gz/131;

    // Convert data to a JSON string
    String payload = "{\"acceleration_x\":" + String(Dax, 3) +
                     ",\"acceleration_y\":" + String(Day, 3) +
                     ",\"acceleration_z\":" + String(Daz, 3) +
                     ",\"gyroscope_x\":" + String(Dgx, 3) +
                     ",\"gyroscope_y\":" + String(Dgy, 3) +
                     ",\"gyroscope_z\":" + String(Dgz, 3) + "}";

    Serial.println("Skiikar....");
    // server address, port and URL
    // Send data via WebSocket
    client.sendTXT(payload);
    client.loop();

    delay(75); // Adjust delay as needed
  }
}
