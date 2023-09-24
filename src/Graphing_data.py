import numpy as np
import matplotlib.pyplot as plt

spo2_data =     np.load("../numpy_saved_data/spo2_data.npy")
pulse_data =    np.load("../numpy_saved_data/pulse_data.npy")
hr_data =       np.load("../numpy_saved_data/hr_data.npy")
resp_data =     np.load("../numpy_saved_data/resp_data.npy")
abp_sys_data =  np.load("../numpy_saved_data/abp_sys_data.npy")
abp_dia_data =  np.load("../numpy_saved_data/abp_dia_data.npy")
abp_mean_data = np.load("../numpy_saved_data/abp_mean_data.npy")

plt_labels = ["SPO2", "PULSE", "HR", "RESP", "ABPsys", "ABPdia", "ABPmean"]

print(len(spo2_data))

combined_data = []

combined_data.append(spo2_data)
combined_data.append(hr_data)
combined_data.append(pulse_data)
combined_data.append(resp_data)
combined_data.append(abp_sys_data)
combined_data.append(abp_dia_data)
combined_data.append(abp_mean_data)

# print(len(combined_data))

correlation_matrix = np.corrcoef(combined_data)

plt.imshow(correlation_matrix, cmap='plasma')

for i in range(correlation_matrix.shape[0]):
    for j in range(correlation_matrix.shape[1]):
        color = "white"
        if(correlation_matrix[i, j] > 0.5):
            color = "black"
        plt.text(j, i, f'{correlation_matrix[i, j]:.2f}', ha='center', va='center', color=color)

# np.save("039_correlation.npy", correlation_matrix)

plt.xticks(range(len(plt_labels)), plt_labels, rotation = 45)
plt.yticks(range(len(plt_labels)), plt_labels)

plt.xlabel("Health Parameter")
plt.ylabel("Health Parameter")

plt.show()