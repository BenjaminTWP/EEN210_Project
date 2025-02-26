import os
import json
from datetime import datetime
from collections import deque
import pandas as pd
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi import WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ServerUtils import *

app = FastAPI()
# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# "mount" a 'static' folder to serve HTML, CSS, JS, and images
app.mount("/static", StaticFiles(directory="./src/static"), name="static")

with open("./src/index.html", "r") as f:
    html = f.read()


class DataProcessor:
    def __init__(self):
        self.data_window = deque()
        self.data_window_length = 0
        self.data_buffer = []
        self.file_path = "./data/data_log.csv"

    def add_window_data(self, data):
        self.data_window.append(data)
        self.data_window_length += 1

    def clear_window(self):
        WINDOW_ENTRIES = 10
        for _ in range(WINDOW_ENTRIES):
            self.data_window.pop()
            self.data_window_length -= 1

    def get_evaluation_data(self):
        eval_data = pd.DataFrame(self.data_window)
        eval_data = eval_data.drop(["timestamp"], axis=1)
        eval_data = eval_data.dropna()
        return eval_data

    def set_filename(self, filename):
        self.file_path = f"./data/{filename}.csv"

    def add_data(self, data):
        self.data_buffer.append(data)

    def save_to_csv(self):
        df = pd.DataFrame(self.data_buffer)
        self.data_buffer = [] #reset buffer

        df.to_csv(
            self.file_path,
            index=False,
            mode="a",
            header=not os.path.exists(self.file_path),
        )
        print(f"Data saved to {self.file_path}")

async def predict_async(data):
    return model.predict(data)

class WebSocketManager:
    def __init__(self):
        self.active_connections = set()
        self.recording = False  # If we decide to record data

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


data_processor = DataProcessor()
model = RuleBasedClassifier(majority_treshold = 0.65)
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
            json_data = json.loads(data)

            if "action" in json_data:
                handle_action_call(websocket_manager, data_processor, json_data)
            else:
                data_processor.add_window_data(json_data)

            if data_processor.data_window_length == 50:
                eval_data_df = data_processor.get_evaluation_data()
                prediction = await predict_async(eval_data_df)
                print(f"RB prediction: {prediction}")

                prediction_message = json.dumps({"type": "prediction", "label": prediction})
                await websocket_manager.broadcast_message(prediction_message)

                if prediction == "fall":
                    print("CALL FOR PATIENT DATA AND SEND TO WEBSERVER")
                    patient_info = handle_patient_information_call()
                    await websocket_manager.broadcast_message(json.dumps(patient_info))

                data_processor.clear_window()

            json_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if websocket_manager.recording:
                data_processor.add_data(json_data)

            sensor_message = json.dumps({"type": "sensor", "data": json_data})
            await websocket_manager.broadcast_message(sensor_message)

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)