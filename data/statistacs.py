import os
import pandas as pd


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

folder_average("./data")


