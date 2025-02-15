import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier

def extract_data_whole():

    X_data = pd.DataFrame(columns=["acceleration_x","acceleration_y","acceleration_z","gyroscope_x","gyroscope_y","gyroscope_z","fall"])

    for folder in os.listdir("./data"):  
        folder_path = os.path.join("./data", folder)
        
        if not os.path.isdir(folder_path):  
            continue

        for file in os.listdir(folder_path):
            label = 0

            temp_df = pd.read_csv(os.path.join(folder_path, file))
            
            temp_df = temp_df.drop(['timestamp'], axis=1)
            #print(temp_df.columns)


            if folder == "fall":
                label = 1
            else:
                label = 0
            
            statistics = temp_df.std().to_frame().T  
            statistics['fall'] = label  
            X_data = pd.concat([X_data, statistics], ignore_index=True)

    X_data["fall"] = X_data["fall"].astype(int)

    return X_data
            
def extract_data():
    X_data = pd.DataFrame(columns=["acceleration_x", "acceleration_y", "acceleration_z", 
                                   "gyroscope_x", "gyroscope_y", "gyroscope_z", "fall"])

    for folder in os.listdir("./data"):  
        folder_path = os.path.join("./data", folder)
        
        if not os.path.isdir(folder_path):  
            continue

        for file in os.listdir(folder_path):
            label = 1 if folder == "fall" else 0

            temp_df = pd.read_csv(os.path.join(folder_path, file))
            temp_df = temp_df.drop(['timestamp'], axis=1)

            # Split into two halves
            first_half = temp_df.iloc[:25]
            second_half = temp_df.iloc[25:]

            for split_df in [first_half, second_half]:
                statistics = split_df.mean().to_frame().T 
                statistics['fall'] = label 
                X_data = pd.concat([X_data, statistics], ignore_index=True)

    X_data["fall"] = X_data["fall"].astype(int)
    return X_data

X_data_df = extract_data()
y_data_df = X_data_df["fall"]

X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X_data_df.drop(["fall"], axis=1), y_data_df, test_size=0.2, random_state=10)

n_trees = np.arange(1, 100, 2)

accuracy = []
recall = []
precision = []
f1 = []

for n in n_trees:
    random_forest_classifier = RandomForestClassifier(n_estimators=n, random_state=10, max_depth=None)
    random_forest_classifier.fit(X_train_df.copy(), y_train_df.copy())
    y_pred_RF = random_forest_classifier.predict(X_test_df.copy())

    accuracy.append(accuracy_score(y_test_df, y_pred_RF))
    recall.append(recall_score(y_test_df, y_pred_RF))
    precision.append(precision_score(y_test_df, y_pred_RF))
    f1.append(f1_score(y_test_df, y_pred_RF))

plt.plot(n_trees, accuracy, label="Accuracy")
plt.plot(n_trees, precision, label="Precision")
plt.plot(n_trees, recall, label="Recall")
plt.plot(n_trees, f1, label="F1")

plt.xlabel("Number of Trees")
plt.ylabel("Scores")
plt.title("Scores vs. Number of Trees")
plt.legend()
plt.grid(True)
plt.show()


n_tree = 20
depth = None
random_forest_classifier = RandomForestClassifier(n_estimators= n_tree, random_state=10, max_depth=depth)

random_forest_classifier.fit(X_train_df, y_train_df)

y_pred_RF = random_forest_classifier.predict(X_test_df)

print(f"Accuracy: {accuracy_score(y_test_df, y_pred_RF)}")
print(f"Recall: {recall_score(y_test_df, y_pred_RF)}")
print(f"Precision: {precision_score(y_test_df, y_pred_RF)}")
print(f"F1: {f1_score(y_test_df, y_pred_RF)}")

cm = confusion_matrix(y_test_df, y_pred_RF)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Malignant Cancer Prediction. \n Random Forest Classifier. Trees = {n_tree}, Depth = {depth}.")
plt.show()