import os
import json
from datetime import datetime

import pandas as pd
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi import WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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
        self.data_buffer = []
        self.file_path = "./data/data_log.csv"  #starting name

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
    model = None
    return model


def predict_label(model=None, data=None):
    # you should modify this to return the label
    if model is not None:
        label = model(data)
        return label
    return 0


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
model = load_model()


@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            #print("hi")

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
