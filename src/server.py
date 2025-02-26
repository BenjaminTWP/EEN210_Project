import os
import json
from datetime import datetime
#from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from collections import deque
import pandas as pd
import uvicorn
from transform import df_vectorized
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi import WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import asyncio

app = FastAPI()
# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the 'static' folder to serve HTML, CSS, JS, and images
app.mount("/static", StaticFiles(directory="./src/static"), name="static")

with open("./src/index.html", "r") as f:
    html = f.read()


class DataProcessor:
    def __init__(self):
        self.data_window = deque()
        self.data_window_length = 0
        self.data_buffer = []
        self.file_path = "./data/data_log.csv"  #starting name

    def add_window_data(self, data):
        self.data_window.append(data)
        self.data_window_length += 1

    def clear_window(self):
        WINDOW_ENTRIES = 10
        for i in range(WINDOW_ENTRIES):
            self.data_window.pop()
            self.data_window_length -= 1

    def get_evaluation_data(self):
        eval_data = pd.DataFrame(self.data_window)
        eval_data = eval_data.drop(["timestamp"], axis=1)
        return eval_data

    def set_filename(self, filename):
        self.file_path = f"./data/{filename}.csv"

    def add_data(self, data):
        self.data_buffer.append(data)

    def save_to_csv(self):
        df = pd.DataFrame(self.data_buffer)
        self.data_buffer = []
        # Append the new row to the existing DataFrame
        df.to_csv(
            self.file_path,
            index=False,
            mode="a",
            header=not os.path.exists(self.file_path),
        )
        print(f"Data saved to {self.file_path}")


data_processor = DataProcessor()
#model = load_model("./model/bestest_model.h5")
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#labels = {0: "fall", 1: "other", 2: "still"}

class RuleBasedClassifier:

    def __init__(self):
        self.predictions_made = deque(maxlen=15)
        self.last_fall_index = None

    def _prepare_prediction(self, data):
        accel_magnitude, gyro_magnitude = df_vectorized(data)
        accel_magnitude_np = np.array(accel_magnitude)
        gyro_magnitude_np = np.array(gyro_magnitude)

        accel_std = np.std(accel_magnitude_np)
        gyro_std = np.std(gyro_magnitude_np)

        print(f"Acceleration STD : {str(accel_std)}. Gyroscope STD : {str(gyro_std)}")
        return accel_std, gyro_std


    
    def predict(self, data):
        accel_std, gyro_std = self._prepare_prediction(data)

        if accel_std <= 0.6 and gyro_std <= 10:# or accel_std <= 0.20 or gyro_std <= 7.5 :
            prediction = "still"
        elif accel_std >= 2.0 and gyro_std >= 25:
            prediction = "fall"
            print("POSSIBLE FALL DETECTED")
            self.last_fall_index = len(self.predictions_made) 
        else:
            prediction = "other"

        self.predictions_made.append(prediction)

        # check oldest fall entry
        if self.last_fall_index is not None and self.last_fall_index == 0:
            # Evaluate predictions after this 'fall'
            predictions_after_fall = list(self.predictions_made)[1:]  # Everything after 'fall'
            still_count = predictions_after_fall.count('still')

            if still_count > len(predictions_after_fall) * 0.65:
                return 'fall'

            self.last_fall_index = None  # reset the fall index

        # update the classificiation 
        if self.last_fall_index is not None:
            self.last_fall_index -= 1

        return prediction

model = RuleBasedClassifier()
#lstm_model = load_model("./model/bestest_model.h5")
#lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


def load_model_nah():
    # you should modify this function to return your model
    ##model = load_model()
    print("Model Loaded Successfully")
    return model


async def predict_async(data):
    return predict_label(model, data)

def predict_label(model, data):
    prediction = model.predict(data)

    #print(f"Fall: {prediction[0][0] * 100} %, Other: {prediction[0][1] * 100} %, Still: {prediction[0][2] * 100} %")
    #print(prediction)
    return prediction

#async def predict_lstm_async(data):
#    return predicted_lstm(lstm_model, data)

def predicted_lstm(model, data):
    eval_data_np = data.astype(np.float32).reshape(1, 50, 1) 
    prediction = model.predict(eval_data_np)
    predicted_label = np.argmax(prediction)
    labels = {0: "fall", 1: "other", 2: "still"}

    print(prediction)
    return labels[predicted_label]

class WebSocketManager:
    def __init__(self):
        self.active_connections = set()
        self.recording = False  # Track recording state

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        print("WebSocket connected")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print("WebSocket disconnected")

    async def broadcast_message(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                # Handle disconnect if needed
                self.disconnect(connection)


websocket_manager = WebSocketManager()

@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()

            # Broadcast the incoming data to all connected clients
            json_data = json.loads(data)

            # use raw_data for prediction. NOT currently used
            raw_data = list(json_data.values())

            if "action" in json_data:
                if json_data["action"] == "start":
                    websocket_manager.recording = True
                    data_processor.set_filename(json_data.get("filename", "data_log"))
                    print("Started recording to:", data_processor.file_path)
                elif json_data["action"] == "stop":
                    websocket_manager.recording = False
                    data_processor.save_to_csv()
                    print("Stopped recording and saved data.")
                continue

            data_processor.add_window_data(json_data)

            if data_processor.data_window_length == 50:
                
                eval_data_df = data_processor.get_evaluation_data()

                #accel_mag, _ = df_vectorized(eval_data_df)
                #lstm_prediction = await predict_lstm_async(accel_mag)
                #print(f"Lstm prediction {lstm_prediction}")

                prediction = await predict_async(eval_data_df)

                print(f"RB prediction: " + prediction)
                data_processor.clear_window()


            ''' Old code for predictions 
            In this line we use the model to predict the labels.
            Right now it only return 0.
            You need to modify the predict_label function to return the true label
            """
            label = predict_label(model, raw_data)
            json_data["label"] = label
            '''
            
            # Add time stamp to the last received data
            json_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if websocket_manager.recording:
                data_processor.add_data(json_data)
            # print the last data in the terminal
            #print(json_data)

            # broadcast the last data to webpage
            await websocket_manager.broadcast_message(json.dumps(json_data))

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
