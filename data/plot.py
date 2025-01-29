import matplotlib.pyplot as plt
import numpy as np

 
all_data = np.loadtxt("./fall_data_20231122_155653.csv",
                 delimiter=";", dtype=str)



labels = all_data[1:-1, 0]

accel_x = all_data[1:-1, 1].astype(float)
accel_y = all_data[1:-1, 2].astype(float)
accel_z = all_data[1:-1, 3].astype(float)

position_x = all_data[1:-1, 4].astype(float)
position_y = all_data[1:-1, 5].astype(float)
position_z = all_data[1:-1, 6].astype(float)


plt.plot(labels, position_x, label="Position X")
plt.plot(labels, position_y, label="Position Y")
plt.plot(labels, position_z, label="Position Z")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()