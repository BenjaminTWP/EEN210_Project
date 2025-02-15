import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the main data folder
data_root = "data"

# Lists to store sequences and labels
sequences = []
labels = []

# Loop through each folder (class label)
for folder_name in os.listdir(data_root):
    folder_path = os.path.join(data_root, folder_name)

    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            if file_name.endswith(".csv"):
                df = pd.read_csv(file_path)

                # Convert all data to numeric
                df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

                # Calculate acceleration magnitude
                df['acceleration_magnitude'] = np.sqrt(
                    df['acceleration_x']**2 + df['acceleration_y']**2 + df['acceleration_z']**2
                )

                # Extract only the magnitude column
                data = df[['acceleration_magnitude']].values

                # Store sequence and binary label
                sequences.append(data)
                labels.append('fall' if folder_name == 'fall' else 'other')

# Convert labels to numerical format
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
encoded_labels = to_categorical(encoded_labels)

# Convert sequences to numpy array
sequences = np.array(sequences, dtype=np.float32)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sequences, encoded_labels, test_size=0.2, random_state=10)

# Define the LSTM model
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(50, 1)))
model.add(LSTM(128, return_sequences=True, activation='tanh'))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=30, mode='max', restore_best_weights=True)

model.fit(X_train, y_train, epochs=60, batch_size=16, validation_data=(X_test, y_test), callbacks=[checkpoint, early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Classification report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
print(classification_report(y_true_classes, y_pred_classes))

# Confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
class_names = label_encoder.classes_
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


def LSTM_model():
    return model, None
