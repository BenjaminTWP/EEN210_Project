import numpy as np
from collections import deque
from transform import df_vectorized


class RuleBasedClassifier:
    def __init__(self, majority_treshold):
        self.predictions_made = deque(maxlen=15)
        self.last_fall_index = None
        self.MAJORITY_TRESHOLD = majority_treshold

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
            prediction = "possible-fall"
            print("POSSIBLE FALL DETECTED")
            self.last_fall_index = len(self.predictions_made) 
        else:
            prediction = "other"

        self.predictions_made.append(prediction)

        if self.last_fall_index is not None and self.last_fall_index == 0:
            predictions_after_fall = list(self.predictions_made)[1:]  # everything after "possible-fall" detected
            still_count = predictions_after_fall.count('still')

            if still_count > len(predictions_after_fall) * self.MAJORITY_TRESHOLD:
                return 'fall'

            self.last_fall_index = None  

        if self.last_fall_index is not None:
            self.last_fall_index -= 1

        return prediction
    

def handle_action_call(websocket_manager, data_processor, json_data):
    if json_data["action"] == "start":
        websocket_manager.recording = True
        data_processor.set_filename(json_data.get("filename", "data_log"))
        print("Started recording to:", data_processor.file_path)
    elif json_data["action"] == "stop":
        websocket_manager.recording = False
        data_processor.save_to_csv()
        print("Stopped recording and saved data.")
    elif json_data["action"] == "email":
        print("SEND EMAIL FROM HERE")
