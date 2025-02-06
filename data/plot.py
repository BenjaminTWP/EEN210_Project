import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

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

def averages(path:str):
    all_data_df = pd.read_csv(path)
    print(all_data_df.describe())

    

#plot("./data/fall/fall_side_v_4.csv")
#verages("./data/still/still_b_1.csv")