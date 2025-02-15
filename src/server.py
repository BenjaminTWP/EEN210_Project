import os
import json
from datetime import datetime
import numpy as np
from collections import deque
from LSTM_TEST import LSTM_model
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
        WINDOW_ENTRIES = 20
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

def load_model():
    # you should modify this function to return your model
    model, scaler = LSTM_model() 
    print("Model Loaded Successfully")
    return model


async def predict_async(data):
    return await asyncio.to_thread(predict_label, model, data)

def predict_label(model, data):
    prediction = model.predict(data)
    print(prediction)
    return np.argmax(prediction)


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
model, scaler = LSTM_model()


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

                # Get acceleration magnitude from df_vectorized()
                acc_magnitude, _ = df_vectorized(eval_data_df)

                # Reshape to (1, 50, 1) because the model expects this shape
                acc_magnitude = np.array(acc_magnitude).reshape(1, 50, 1)

                # Predict
                prediction = await predict_async(acc_magnitude)
                predicted_label = 'fall' if prediction == 0 else 'other'

                print(f"Predicted Label: {predicted_label}")
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
