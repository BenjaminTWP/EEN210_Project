import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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


def transformation_test():
    path = "./data"
    folder_stats = {}  # Store variance for each folder

    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)

            if "timestamp" in df.columns:
                df = df.drop(columns=["timestamp"])

            accel_magnitudes, gyro_magnitudes = df_vectorized(df)

            accel_variance = pd.Series(accel_magnitudes).std()
            gyro_variance = pd.Series(gyro_magnitudes).std()

            if folder not in folder_stats:
                folder_stats[folder] = []

            folder_stats[folder].append((accel_variance, gyro_variance))

    # Plotting
    plt.figure()
    colors = plt.cm.get_cmap("tab10")

    for idx, (folder, variances) in enumerate(folder_stats.items()):
        for accel_var, gyro_var in variances:
            plt.scatter(accel_var, gyro_var, color=colors(idx), label=folder)

    # Avoid duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys())

    plt.xlabel("Acceleration")
    plt.ylabel("Gyroscope")
    plt.title("Acceleration vs Gyroscope")
    plt.grid(True)
    plt.show()

transformation_test()

def plot_transformation(path:str):
    all_data_df = pd.read_csv(path)
    timestamp_removed = all_data_df.drop(["timestamp"], axis=1)

    print(timestamp_removed)

    acceleration_transformed = []
    gyroscope_transformed = []

    for _, row in timestamp_removed.iterrows():
        acceleration_transformed.append(transformation(row["acceleration_x"], row["acceleration_y"], row["acceleration_z"]))
        gyroscope_transformed.append(transformation(row["gyroscope_x"], row["gyroscope_y"], row["gyroscope_z"]))

    plt.plot(acceleration_transformed, label= "Acceleration Vector")
    #plt.plot(gyroscope_transformed, label="Gyroscope Vector")
    plt.title(path)
    plt.legend()
    plt.grid()
    plt.show()

#plot_transformation("./data/fall/fall_back_v_1.csv")