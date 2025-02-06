import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot(path:str):
    all_data_df = pd.read_csv(path)
    all_data_df = all_data_df.iloc[0:-1, 0:-1].astype(float)
    all_data = all_data_df.to_numpy()

    accel_x = all_data[1:-1, 0].astype(float)
    accel_y = all_data[1:-1, 1].astype(float)
    accel_z = all_data[1:-1, 2].astype(float)

    position_x = all_data[1:-1, 3].astype(float)
    position_y = all_data[1:-1, 4].astype(float)
    position_z = all_data[1:-1, 5].astype(float)

    plt.plot(position_x, label="Position X")
    plt.plot(position_y, label="Position Y")
    plt.plot(position_z, label="Position Z")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.plot(accel_x, label="Acc X")
    plt.plot(accel_y, label="Acc Y")
    plt.plot(accel_z, label="Acc Z")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def extract_data(path: str, column1, column2, column3):
    data = [[], [], []]  
    labels = []

    for folder in os.listdir(path):  
        folder_path = os.path.join(path, folder)
        if not os.path.isdir(folder_path):  
            continue

        #print(f"Processing: {folder_path}")

        file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')] 
        for file in file_names:
            df = pd.read_csv(os.path.join(folder_path, file))
            means = df[[column1, column2, column3]].std()  #### WE CHANGE HERE, .std, .mean, .var
            labels.append(folder)

            data[0].append(means[column1])
            data[1].append(means[column2])
            data[2].append(means[column3])

    return data, labels

def plot_3d(data, labels, xlabel, ylabel, zlabel, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = list(set(labels))
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

    label_color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    for i in range(len(labels)):
        ax.scatter(
            data[0][i],
            data[1][i],
            data[2][i],
            color=label_color_map[labels[i]],
            label=labels[i] if labels[:i].count(labels[i]) == 0 else "",
            s=50
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    ax.legend(loc='best', fontsize='small', markerscale=0.7)
    plt.show()


matrix, labels = extract_data("./data", "acceleration_x", "acceleration_y", "acceleration_z")
plot_3d(matrix, labels, "acceleration_x", "acceleration_y", "acceleration_z", "Acceleration")

matrix, labels = extract_data("./data", "gyroscope_x", "gyroscope_y", "gyroscope_z")
plot_3d(matrix, labels, "gyroscope_x", "gyroscope_y", "gyroscope_z", "Gyroscope")


#plot("./data/walk/walk_v_4.csv")
#verages("./data/still/still_b_1.csv")