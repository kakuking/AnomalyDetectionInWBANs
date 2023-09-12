import matplotlib.pyplot as plt
import numpy as np

def convert_to_float(value):
    try:
        return float(value)
    except ValueError:
        return None

base = "physionet.org/files/mimicdb/1.0.0/039/"
lower = 1
upper = 200

data = {
    'SpO2': [],
    'PULSE': [],
    'HR': [],
    'RESP': [],
    'ABP': [],
}

for i in range(1, 201):
    filename = f'039{str(i).zfill(5)}.txt' 
    print(filename)

    with open(base + filename, 'r') as file:
        next(file)
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                category, *values = parts
                category = category.rstrip(" ")
                values = [convert_to_float(value) for value in values]
                if category not in data:
                    data[category] = []
                data[category].append(values)

# Now you can access the data in separate lists like this:
spo2_data = np.array(data['SpO2'])
pulse_data = np.array(data['PULSE'])
hr_data =  np.array(data['HR'])
resp_data = np.array(data['RESP'])
abp_data = np.array(data['ABP'])
abp_sys_data = np.array([])
abp_dia_data = np.array([])
abp_mean_data = np.array([])

for row in abp_data:
    if(len(row) < 3):
        continue
    abp_mean_data = np.append(abp_mean_data, row[0])
    abp_sys_data = np.append(abp_sys_data, row[0])
    abp_dia_data = np.append(abp_dia_data, row[0])

np.save("spo2_data.npy", spo2_data)
np.save("pulse_data.npy", pulse_data)
np.save("hr_data.npy", hr_data)
np.save("resp_data.npy", resp_data)
np.save("abp_sys_data.npy", abp_sys_data)
np.save("abp_dia_data.npy", abp_dia_data)
np.save("abp_mean_data.npy", abp_mean_data)




plt.plot(spo2_data, label = "SPO2")
plt.plot(pulse_data, label = "PULSE")
plt.plot(hr_data, label = "HR")
plt.plot(resp_data, label = "RESP")
plt.plot(abp_mean_data, label = "ABPmean")
plt.plot(abp_dia_data, label = "ABPdia")
plt.plot(abp_sys_data, label = "ABPsys")

plt.grid()
plt.ylim(-50, 400)
plt.legend(loc="upper right")
plt.title('ABP')

plt.tight_layout()
plt.show()