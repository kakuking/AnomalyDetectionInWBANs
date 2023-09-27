import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Self-Explanatory
def convert_to_float(value):
    return float(value)

# Save the numpy arrays
def saveArrays():
    np.save("../numpy_saved_data/spo2_data.npy",        np.array(spo2_data).ravel())
    np.save("../numpy_saved_data/pulse_data.npy",       np.array(pulse_data).ravel())
    np.save("../numpy_saved_data/hr_data.npy",          np.array(hr_data).ravel())
    np.save("../numpy_saved_data/resp_data.npy",        np.array(resp_data).ravel())
    np.save("../numpy_saved_data/abp_sys_data.npy",     np.array(abp_sys_data).ravel())
    np.save("../numpy_saved_data/abp_dia_data.npy",     np.array(abp_dia_data).ravel())
    np.save("../numpy_saved_data/abp_mean_data.npy",    np.array(abp_mean_data).ravel())

# base folder
base =  "../physionet.org/files/mimicdb/1.0.0/039/"

data = {
    'SpO2':     [],
    'PULSE':    [],
    'HR':       [],
    'RESP':     [],
    'ABP':      [],
}

# loop through all files and parse the values
for i in tqdm(range(1, 466)):
    filename = f'039{str(i).zfill(5)}.txt' 

    # Open the file and see
    with open(base + filename, 'r') as file:

        # Skip the first line (data)
        next(file)

        for line in file:

            # Parse the line
            parts = line.strip().split('\t')

            # if more than 2 parts (line is not invalid)
            if len(parts) >= 2:

                # split it into parts
                category, *values = parts

                # remove trailing zeros
                category = category.rstrip(" ")

                # Convert values to number
                values = [convert_to_float(value) for value in values]

                if category == "ABP" and len(values) < 3:
                    data["SpO2"].pop()
                    data["PULSE"].pop()
                    data["HR"].pop()
                    data["RESP"].pop()
                    continue
                
                # add it to arrays
                if category not in data:
                    data[category] = []
                data[category].append(values)

# Separate it into arrays
spo2_data =     data['SpO2']
pulse_data =    data['PULSE']
hr_data =       data['HR']
resp_data =     data['RESP']
abp_data =      data['ABP']
abp_sys_data =  []
abp_dia_data =  []
abp_mean_data = []

unneeded = 0

# Split abp into the three components
for row in abp_data:
    if(len(row) < 3):
        unneeded += 1
        continue
    abp_mean_data.append(row[0])
    abp_sys_data.append(row[1])
    abp_dia_data.append(row[2])

# Save the numpy arrays
saveArrays()

# PyPlot stuff
plt.plot(data["SpO2"],  label = "SPO2")
plt.plot(data["PULSE"], label = "PULSE")
plt.plot(data["HR"],    label = "HR")
plt.plot(data["RESP"],  label = "RESP")
plt.plot(abp_mean_data, label = "ABPmean")
plt.plot(abp_dia_data,  label = "ABPdia")
plt.plot(abp_sys_data,  label = "ABPsys")

plt.grid()
plt.ylim(-50, 400)
plt.legend(loc="upper right")
plt.title('ABP')

plt.ioff()
plt.tight_layout()

plt.show()
