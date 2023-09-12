import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../Database_UNEDITED/CHARTEVENTS.csv')

ABPsys = 220050
ABPmean = 220051
ABPdia = 220052
sp02 = 220277
temp = 223762
resp = 220210
hr = 220045

filteredABPsys = df[df['itemid'] == ABPsys]

filteredABPsys.loc[:, 'charttime'] = pd.to_datetime(filteredABPsys['charttime'], format='%Y-%m-%d %H:%M')
values_to_plot = filteredABPsys['value']

# Create a time series plot
plt.figure(figsize=(12, 6))
plt.plot(filteredABPsys['charttime'], values_to_plot, marker='o', linestyle='-', color='b', label='Values')
plt.xlabel('Time')
plt.ylabel('ABP systolic')
plt.title('ABP systolic vs. Time')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Display the plot or save it to a file
plt.show()