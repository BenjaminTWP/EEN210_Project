import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout # type: ignore
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input # type: ignore
from sklearn.preprocessing import LabelEncoder
from transform import df_vectorized
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def data_and_label_extraction():
    data_root = "data"
    sequences = []
    labels = []

    for folder_name in os.listdir(data_root):
        folder_path = os.path.join(data_root, folder_name)

        if os.path.isdir(folder_path):

            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                if file_name.endswith(".csv"):
                    df = pd.read_csv(file_path)
                    
                    data, _ = df_vectorized(df)
                    sequences.append(data)

                    labels.append('fall' if folder_name == 'fall' else 'other')

    sequences = np.array(sequences, dtype=np.float32)

    return labels, sequences


def encode_labels(labels:list):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    encoded_labels = to_categorical(encoded_labels)
    return encoded_labels, label_encoder


def configure_model(y_train):
    model = Sequential([
        Input(shape=(50, 1)),
        Masking(mask_value=0.0),
        LSTM(128, return_sequences=True, activation='tanh'),
        Dropout(0.3),
        LSTM(64, activation='tanh'),
        Dropout(0.3),
        Dense(y_train.shape[1], activation='softmax')
    ])
    return model


def train_model(model, X_train, X_test, y_train, y_test):

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('./model/best_val_model.h5', monitor='val_loss', save_best_only=True, mode='max')

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=30, mode='max', restore_best_weights=True)

    model.fit(X_train, y_train, epochs=60, batch_size=16, validation_data=(X_test, y_test), callbacks=[checkpoint, early_stopping])


def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    return y_pred_classes, y_true_classes

def plot_confusion_matrix(y_true_classes, y_pred_classes, label_encoder):
    
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    class_names = label_encoder.classes_
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def model_implementation():
    labels, sequences = data_and_label_extraction()
    encoded_labels, label_encoder = encode_labels(labels)

    X_train, X_test, y_train, y_test = train_test_split(sequences, encoded_labels, test_size=0.2, random_state=10)

    model = configure_model(y_train)

    train_model(model, X_train, X_test, y_train, y_test)

    y_pred, y_true = evaluate_model(model, X_test, y_test)

    plot_confusion_matrix(y_pred, y_true, label_encoder)


model_implementation()