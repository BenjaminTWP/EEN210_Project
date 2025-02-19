import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout # type: ignore
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.models import load_model # type: ignore
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

                    if folder_name == "fall":
                        labels.append("fall")
                    elif folder_name == "still":
                        labels.append("still")
                    else:
                        labels.append("other")

    sequences = np.array(sequences, dtype=np.float32)

    return labels, sequences


def encode_labels(labels:list):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    encoded_labels = to_categorical(encoded_labels)
    print(label_encoder.classes_)
    return encoded_labels, label_encoder


def configure_model(y_train):
    model = Sequential([
        Input(shape=(50, 1)),
        LSTM(64, activation='tanh'),
        Dense(y_train.shape[1], activation='softmax')
    ])
    return model



def train_model(model, X_train, X_test, y_train, y_test):

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #checkpoint = ModelCheckpoint('./model/best_val_model.h5', monitor='val_loss', save_best_only=True, mode='max')

    #model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test), callbacks=[checkpoint])


    from sklearn.utils import class_weight

    labels_numeric = np.argmax(y_train, axis=1)
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels_numeric), y=labels_numeric)
    class_weights = dict(enumerate(weights))

    model.fit(X_train, y_train, epochs=60, batch_size=8, validation_data=(X_test, y_test), callbacks=[], class_weight=class_weights)



def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    return y_pred_classes, y_true_classes

def plot_confusion_matrix(y_true_classes, y_pred_classes, label_encoder):
    print("True class distribution:", np.bincount(y_true_classes))
    print("Predicted class distribution:", np.bincount(y_pred_classes))

    
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
    label_encoder.classes_




    #X_train, X_test, y_train, y_test = train_test_split(sequences, encoded_labels, test_size=0.2, random_state=10)

    #model = configure_model(y_train)

    #train_model(model, X_train, X_test, y_train, y_test)

    #best_model = load_model("./model/best_val_model.h5")
    #best_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #model.save("./model/best_val_model.h5")


    #y_pred, y_true = evaluate_model(model, X_test, y_test)

    #plot_confusion_matrix(y_pred, y_true, label_encoder)


model_implementation()

def temp():
    labels, sequenes = data_and_label_extraction()
    print(sequenes.shape)

    std_falls = []
    std_other = []
    std_still = []

    for i in range(len(sequenes)):
        if labels[i] == "fall":
            std_falls.append(np.std(sequenes[i]))
        elif labels[i] == "other":
            std_other.append(np.std(sequenes[i]))
        else:
            std_still.append(np.std(sequenes[i]))
        

    fall_mean_std = np.mean(std_falls)
    other_mean_std = np.mean(std_other)
    still_mean_std = np.mean(std_still)

    fall_std = np.std(std_falls)
    other_std = np.std(std_other)
    still_std = np.std(std_still)
    plot_distributions(fall_mean_std, fall_std, other_mean_std, other_std, still_mean_std, still_std)


from scipy.stats import norm

def plot_distributions(fall_mean, fall_std, other_mean, other_std, still_mean, still_std):
    categories = ['Fall', 'Other', 'Still']
    means = [fall_mean, other_mean, still_mean]
    stds = [fall_std, other_std, still_std]
    colors = ['red', 'blue', 'green']

    fig, ax = plt.subplots(figsize=(10, 6))
    x_min = min(fall_mean - 3*fall_std, other_mean - 3*other_std, still_mean - 3*still_std)
    x_max = max(fall_mean + 3*fall_std, other_mean + 3*other_std, still_mean + 3*still_std)
    x = np.linspace(x_min, x_max, 500)

    for mean, std, color, label in zip(means, stds, colors, categories):
        ax.plot(x, norm.pdf(x, mean, std), label=label, color=color, alpha=0.7)

    ax.set_title('Standard Distributions for Fall, Other, and Still')
    ax.set_xlabel('Standard Deviation Values')
    ax.set_ylabel('Probability Density')
    ax.set_ylim(0, 0.2)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.show()

#temp()