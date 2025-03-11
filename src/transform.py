import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random


def folder_average(path):
    for folder in os.listdir(path):  
        folder_path = os.path.join(path, folder)
        
        if not os.path.isdir(folder_path):  
            continue

        print(f"Processing folder: {folder}")

        file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')] 

        df = pd.concat([pd.read_csv(os.path.join(folder_path, f)) for f in file_names], ignore_index=True)

        means = df.describe().loc["mean"]
        print(means)


#folder_average("./data")

def transformation(a, b, c):
    return np.sqrt(a ** 2 + b ** 2 + c ** 2)

def calculate_magnitude(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)

def df_vectorized(df):
    accel_magnitudes = []
    gyro_magnitudes = []
    for _, row in df.iterrows():
        accel_magnitude = calculate_magnitude(row["acceleration_x"], row["acceleration_y"], row["acceleration_z"])
        gyro_magnitude = calculate_magnitude(row["gyroscope_x"], row["gyroscope_y"], row["gyroscope_z"])
            
        accel_magnitudes.append(accel_magnitude)
        gyro_magnitudes.append(gyro_magnitude)

    return np.array(accel_magnitudes), np.array(gyro_magnitudes)


def extract_transformation_data():
    path = "./data"
    folder_stats = {}  # storing statistics for each category

    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if not os.path.isdir(folder_path):
            continue

        category = folder if folder in ["still", "fall"] else "other"

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)

            if "timestamp" in df.columns:
                df = df.drop(columns=["timestamp"])

            accel_magnitudes, gyro_magnitudes = df_vectorized(df)

            accel_variance = pd.Series(accel_magnitudes).std()
            gyro_variance = pd.Series(gyro_magnitudes).std()

            if category not in folder_stats:
                folder_stats[category] = []

            folder_stats[category].append((accel_variance, gyro_variance, file))
    
    return folder_stats

def split_data(folder_stats, train_ratio=0.8):
    train_data = {}
    test_data = {}

    for category, values in folder_stats.items():
        random.shuffle(values)
        split_idx = int(len(values) * train_ratio)
        train_data[category] = values[:split_idx]
        test_data[category] = values[split_idx:]

    return train_data, test_data

def plot_transformation_data(train_data):
    plt.figure()
    colors = plt.cm.get_cmap("tab10")

    for idx, (folder, variances) in enumerate(train_data.items()):
        for accel_var, gyro_var, filename in variances:
            plt.scatter(accel_var, gyro_var, color=colors(idx), label=folder)
            #plt.text(accel_var, gyro_var, filename, fontsize=8, alpha=0.7)  # Add filename as label

    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys())

    plt.xlabel("Acceleration Magnitude STD")
    plt.ylabel("Gyroscope Magnitude STD")
    plt.grid(True)
    plt.show()


folder_stats = extract_transformation_data()
train_data, test_data = split_data(folder_stats)
#plot_transformation_data(train_data)


def plot_transformation(path:str, title):
    all_data_df = pd.read_csv(path)
    timestamp_removed = all_data_df.drop(["timestamp"], axis=1)

    print(timestamp_removed)

    acceleration_transformed = []
    gyroscope_transformed = []

    for _, row in timestamp_removed.iterrows():
        acceleration_transformed.append(transformation(row["acceleration_x"], row["acceleration_y"], row["acceleration_z"]))
        gyroscope_transformed.append(transformation(row["gyroscope_x"], row["gyroscope_y"], row["gyroscope_z"]))

    plt.figure(figsize=(5.5,4))
    print(np.std(acceleration_transformed[1:75]))
    plt.plot(acceleration_transformed[0:75], label= "Acceleration Magnitude")
    #plt.plot(gyroscope_transformed[0:75], label="Acceleration Magnitude")
    plt.title(title)
    plt.legend()
    plt.xlabel("Number of time entries")
    plt.ylim((8.9,9.3))

    plt.grid()
    plt.show()

plot_transformation("./data/moving_when_starting.csv", "Still sensor readings when sensor \n was intialized with movement")
plot_transformation("./data/still_when_starting.csv", "Still sensor readings when sensor \n was intialized without movement")