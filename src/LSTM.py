import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



# Path to the main data folder
data_root = "data"

# Lists to store sequences and labels
sequences = []
labels = []
file_count = 0  

# Maximum sequence length tracking
max_seq_length = 0

# Loop through each folder (class label)
for folder_name in os.listdir(data_root):
    folder_path = os.path.join(data_root, folder_name)

    if os.path.isdir(folder_path):
        print(f"Loading data from: {folder_path}")

        # Loop through each CSV file
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            if file_name.endswith(".csv"):
                df = pd.read_csv(file_path)

                # Convert all data to numeric
                df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
                df = df.iloc[:, :-1]

                # Extract all features
                data = df.values  # Shape: (time_steps, features)
                #print(data.shape)

                # Track max sequence length
                max_seq_length = max(max_seq_length, data.shape[0])

                # Store sequence and label
                sequences.append(data)
                labels.append(folder_name)
                file_count += 1  

print(f"Total files loaded: {file_count}")

# Convert labels to numerical format
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
encoded_labels = to_categorical(encoded_labels)

# Padding sequences to ensure uniform shape
#sequences = pad_sequences(sequences, maxlen=max_seq_length, dtype='float32', padding='post', truncating='post')

print(label_encoder.classes_) 

#print(f"Final dataset shape: {sequences.shape}")  # (num_samples, max_time_steps, num_features)
#print(f"Labels shape: {encoded_labels.shape}")  # (num_samples, num_classes)



# Split data into training and testing sets
sequences = np.array(sequences, dtype=np.float32)
X_train, X_test, y_train, y_test = train_test_split(sequences, encoded_labels, test_size=0.2, random_state=42)

#print(f"Training set: {X_train.shape}, {y_train.shape}")
#print(f"Testing set: {X_test.shape}, {y_test.shape}")

# Define the LSTM model
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(50, X_train.shape[2])))  # Fixed input length

model.add(LSTM(128, return_sequences=True, activation='tanh'))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Classification report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
print(classification_report(y_true_classes, y_pred_classes))


def confusion():
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    # Get class names from label encoder
    class_names = label_encoder.classes_

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()




def LSTM_model():
    return model, None