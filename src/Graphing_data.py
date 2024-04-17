import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pre_process():
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

def autoencoder_training():
    # Assuming your CSV file is named 'your_data.csv'
    csv_file_path = './autoencoder.csv'

    # Read data from CSV file
    data = pd.read_csv(csv_file_path)

    # Extract relevant columns
    num_epochs = data['NUM_EPOCHS']
    num_anomalies = data['NUM_ANOMALIES']
    mae = data['MAE']
    mse = data['MSE']

    # Plotting
    # plt.plot(num_epochs, num_anomalies, label='Anomalies', marker='o', linestyle='-')
    plt.plot(num_epochs, mae, label='MAE', marker='o', linestyle='-')
    # plt.plot(num_epochs, mse, label='MSE', marker='o', linestyle='-')
    # plt.title('Number of Epochs vs Number of Anomalies')
    plt.title('Number of Epochs vs MAE')
    # plt.title('Number of Epochs vs MSE')
    # plt.xlabel('Number of Epochs')
    # plt.ylabel('Number of Anomalies')
    plt.ylabel('MAE')
    plt.ylabel('MSE')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()
    
def lstm_training():
    # Assuming your CSV file is named 'your_data.csv'
    csv_file_path = './lstm.csv'

    # Read data from CSV file
    data = pd.read_csv(csv_file_path)

    # Extract relevant columns
    num_epochs = data['Num_Epochs']
    accuracy = data['accuracy']
    precision = data['precision']
    recall = data['recall']
    f1_score = data['f1_score']

    # Plotting
    # plt.plot(num_epochs, accuracy, label='Accuracy', marker='o', linestyle='-')
    # plt.plot(num_epochs, precision, label='Precision', marker='o', linestyle='-')
    # plt.plot(num_epochs, recall, label='Recall', marker='o', linestyle='-')
    plt.plot(num_epochs, f1_score, label='F1 Score', marker='o', linestyle='-')

    # plt.title('Number of Epochs vs Accuracy')
    # plt.title('Number of Epochs vs Precision')
    # plt.title('Number of Epochs vs Recall')
    plt.title('Number of Epochs vs F1 Score')

    # plt.ylabel('Accuracy')
    # plt.ylabel('Precision')
    # plt.ylabel('Recall')
    plt.ylabel('F1 Score')
    
    plt.xlabel('Number of Epochs')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()
    
    
lstm_training()
# autoencoder_training()
