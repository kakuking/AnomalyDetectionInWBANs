from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential, load_model

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import csv

# Load the database and correlate it
def load_and_correlate_dataset(INDEX_TO_CHECK)-> [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    correlation_matrix: np.ndarray = np.load(base_path + "039_correlation.npy")
    encoded_data:       np.ndarray = np.load(base_path + "039_encoded_dataset.npy")
    combined_data:      np.ndarray = np.load(base_path + "039_combined_dataset.npy")
    labels:             np.ndarray = np.load(base_path + "039_LSTM_labels.npy")

    labels = labels.reshape(-1, 1)

    correlated_data: np.ndarray = np.sum(combined_data * correlation_matrix[INDEX_TO_CHECK].T,
                             axis=1,
                             keepdims=True)

    print(f"Correlation matrix shape:       {correlation_matrix.shape}")
    print(f"Dataset shape:                  {combined_data.shape}")
    print(f"Labels shape:                   {labels.shape}")
    print(f"Correlated data shape:          {correlated_data.shape}")
    print(f"Encoded dataset shape:          {encoded_data.shape}")

    return correlation_matrix, combined_data, correlated_data, labels, encoded_data

# Find the anomaly scores
def find_anomaly_scores(correlated_data) -> np.ndarray:
    correlated_mean = np.mean(correlated_data)
    correlated_std = np.std(correlated_data)
    anomaly_scores = np.abs(correlated_data - correlated_mean)/correlated_std

    # print(f"Correlated data mean:           {correlated_mean}")
    # print(f"Correlated data std:            {correlated_std}")
    # print(f"Anomaly scores shape:           {anomaly_scores.shape}")

    return anomaly_scores

# Find anomalous_indices
def find_contextual_anomalies(anomaly_scores, labels) -> np.ndarray:
    anomaly_std = np.std(anomaly_scores)
    indices_with_label = np.where(labels == 1)[0]

    anomalous_indices = np.where(anomaly_scores > anomaly_std)[0]

    print(f"Anomalous indices before filter:                {anomalous_indices.shape}")

    anomalous_indices = np.intersect1d(anomalous_indices, indices_with_label)

    print(f"Anomalous indices filtered by point anomaly:    {anomalous_indices.shape}")

    return anomalous_indices

# Takes indices of anomalies and converts into labels
def create_anomalous_labels() -> np.ndarray:
    LSTM_labels = np.zeros(combined_data.shape[0])
    LSTM_labels[anomalous_indices] = 1
    
    return LSTM_labels

# Takes anomalous labels that are for each row (7x1) and converts to each encoding
# Each 2x2x64 encoding is the encoding of a 7x7 piece of data, hence we check 
# if there is any anomaly in the 7x7 range and if there is, it is labelled true
def contract_anomalous_labels(labels) -> np.ndarray:
    num = len(labels) // 7
    temp = labels.reshape((num, 7))
    contracted_labels = np.sum(temp, axis=1) >= 1

    contracted_labels = contracted_labels.astype(int)

    print(f"Anomalous indices after contraction from 7x7 to 7x1: {contracted_labels.shape}")

    return contracted_labels

# Creates sequences for encoded data to feed to LSTM
# Also creates labels for it
def reshape_encoded_dataset(encoded_dataset, contracted_anomalous_labels, sequence_length) -> [np.ndarray, np.ndarray]:
    num_samples, sample_dim2, sample_dim3, sample_dim4 = encoded_dataset.shape
    num_seq = num_samples - sequence_length + 1
    
    sequences = np.zeros((num_seq, sequence_length, sample_dim2, sample_dim3, sample_dim4))
    sequence_labels = np.zeros(num_seq)

    for i in range(num_seq):
        sequences[i] = encoded_dataset[i:i+sequence_length]
        sequence_labels[i] = contracted_anomalous_labels[i+sequence_length-1]

    print(f"Created sequences from encoded data: {sequences.shape}")
    print(f"Labels for sequences: {sequence_labels.shape}")

    return sequences, sequence_labels

# Splits data into train test validate
def split_into_train_test_val(encoded_dataset, dataset_labels, test_ratio, val_ratio):
    X_train, X_temp, Y_train, Y_temp = train_test_split(encoded_dataset, dataset_labels, test_size = test_ratio + val_ratio)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=test_ratio/(val_ratio + test_ratio))

    print(f"Split sequences of encoded data into train {X_train.shape}, test {X_test.shape}, val {X_val.shape}")

    return X_train, X_test, X_val, Y_train, Y_test, Y_val

# Reshape any dataset from (A, 7, 2, 2, 64) ----> (A, 7, 256)
def reshape_dataset(dataset, sequence_length):
    dataset = dataset.reshape(dataset.shape[0], sequence_length, -1)

    return dataset

# Reshapes X_train, X_test, X_val using reshape_dataset
def reshape_train_test_val(X_train, X_test, X_val, sequence_length):
    X_train = reshape_dataset(X_train, sequence_length)   
    X_test = reshape_dataset(X_test, sequence_length)   
    X_val = reshape_dataset(X_val, sequence_length)

    return X_train, X_test, X_val

# Creates LSTM model
def create_LSTM_model(input_shape) -> Sequential:

    '''
        V1:
        LSTM(64, input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')

        V2: 
        LSTM(64, input_shape=input_shape, return_sequences=True),
        LSTM(64, return_sequences=True),
        LSTM(64),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    '''


    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        LSTM(64, return_sequences=True),
        LSTM(64),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # print("Created model")

    return model

# Trains prev created model
def train_model(model, dataset, labels, val_dataset, val_labels, num_epochs, batch_size):
    model.fit(dataset, labels,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    shuffle=True, 
                    validation_data=(val_dataset, val_labels),
                    )

    print("Model trained")

    model.summary()

    model.save(model_path)

    print("Model saved")

    return model

# Do binary predictions
def binary_predict(model, input):
    predictions = model.predict(input, verbose=0)
    binary_predictions = (predictions > 0.5).astype(int)

    return binary_predictions, predictions

def predict_and_metrics(model, inpu, actual_labels):
    Y_predicted, _ = binary_predict(model, inpu)

    print(f"predicted labels sum:   {np.sum(Y_predicted)}")
    print(f"predicted labels shape: {Y_predicted.shape}")

    test_loss, test_accuracy = LSTM_model.evaluate(reshaped_encoded_sequences, actual_labels, verbose=0)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

# Create graphs for contextual anomalies
def predict_and_graph(model, input, INDEX_TO_CHECK, point_labels, combined_data, AS_contextual_anomalies, AS_contracted):
    point_anomaly_indices = np.where(point_labels == 1)[0]

    predicted_contextual, _ = binary_predict(model, input)

    predicted_contextual_indices = 7 * np.where(predicted_contextual == 1)[0]
    calculated_contracted_indices = 7 * np.where(AS_contracted == 1)[0]

    # predicted_contextual_indices = np.column_stack((predicted_contextual_indices, predicted_end_indices))

    print(f"Calculated contextual anomaly indices shape:                {AS_contextual_anomalies.shape}")
    print(f"Calculated contracted contextual anomaly indices shape:     {calculated_contracted_indices.shape}")
    print(f"Predicted Contextual anomaly indices shape:                 {predicted_contextual_indices.shape}")
    print(f"Point anomaly indices shape:                                {point_anomaly_indices.shape}")


    plt.plot(combined_data[:, INDEX_TO_CHECK], label="Value of sensor")
    plt.plot(predicted_contextual_indices, combined_data[predicted_contextual_indices, INDEX_TO_CHECK], 'yo', label = "Predicted Contextual anomalies")
    plt.plot(calculated_contracted_indices, combined_data[calculated_contracted_indices, INDEX_TO_CHECK], 'ro', label = "Calculated Contextual anomalies", markersize=2)
    # plt.plot(AS_contextual_anomalies, combined_data[AS_contextual_anomalies, INDEX_TO_CHECK], 'ro', label = "AS Contextual anomalies", markersize=2)

    plt.xlabel("Time Step")
    plt.ylabel("spO2 reading")
    
    plt.grid()
    plt.legend(loc="upper right")

    plt.ioff()
    plt.tight_layout()

    plt.show()

def predict_and_graph_AS(model, input, Anomaly_scores, AS_contracted):
    predictions, raw_values = binary_predict(model, input)
    raw_values = np.repeat(raw_values, 7, axis=0)

    predicted_contextual_indices = 7 * np.where(predictions == 1)[0]
    calculated_contracted_indices = 7 * np.where(AS_contracted == 1)[0]
    
    print(f"Predicted Number of anomalies =     {predicted_contextual_indices.shape}")
    print(f"Calculated Number of anomalies =    {calculated_contracted_indices.shape}")
    
    plt.plot(Anomaly_scores, label="Anomaly Scores as Calculated")
    # plt.plot(raw_values, label="Raw predicted values of anomaly scores")
    plt.plot(predicted_contextual_indices, Anomaly_scores[predicted_contextual_indices], 'ro', label="Predicted Contextual Anomalies")
    plt.plot(calculated_contracted_indices, Anomaly_scores[calculated_contracted_indices], 'bo', label="AS calculated Contextual Anomalies", markersize=2)
    plt.grid()
    plt.legend(loc="upper right")
    plt.show()

def trainAndVsAccAndAll(model, dataset, labels, num_epochs, val_dataset, val_labels, batch_size, data, y_true):
    model.fit(dataset, labels,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    shuffle=True, 
                    validation_data=(val_dataset, val_labels),
                    verbose=2
                    )

    y_pred, _ = binary_predict(model, data)
    
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    with open(csv_file_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([str(num_epochs),str(acc), str(precision), str(recall), str(f1)])

'''
#################################################################################################################################################
'''

base_path = "../numpy_saved_data/"
model_path = "../models/LSTM_modelV2.h5"
csv_file_path = "./lstm.csv"
INDEX_TO_CHECK = 0

'''
spo2_data ========== 0
pulse_data ========= 1
hr_data ============ 2
resp_data ========== 3
abp_sys_data ======= 4
abp_dia_data ======= 5
abp_mean_data ====== 6
'''

correlation_matrix, combined_data, correlated_data, point_labels, encoded_data = load_and_correlate_dataset(INDEX_TO_CHECK)
anomaly_scores = find_anomaly_scores(correlated_data)
anomalous_indices = find_contextual_anomalies(anomaly_scores, point_labels)
anomalous_labels = create_anomalous_labels()
contracted_anomalous_labels = contract_anomalous_labels(anomalous_labels)

# Choose a sequene length of 7
SEQUENCE_LENGTH = 7
encoded_sequences, sequence_labels = reshape_encoded_dataset(encoded_data, contracted_anomalous_labels, SEQUENCE_LENGTH)

X_train, X_test, X_val, Y_train, Y_test, Y_val = split_into_train_test_val(encoded_sequences, sequence_labels, 0.1, 0.2)
X_train, X_test, X_val = reshape_train_test_val(X_train, X_test, X_val, SEQUENCE_LENGTH)

num_epochs = 50
batch_size = 128
reshaped_encoded_sequences = reshape_dataset(encoded_sequences, SEQUENCE_LENGTH)

LSTM_model = create_LSTM_model(X_train.shape[1:])

# LSTM_model = train_model(LSTM_model, X_train, Y_train, X_val, Y_val, num_epochs, batch_size)
LSTM_model = load_model(model_path) 

predict_and_graph(LSTM_model, reshaped_encoded_sequences, INDEX_TO_CHECK, point_labels, combined_data, anomalous_indices, contracted_anomalous_labels)
# predict_and_metrics(LSTM_model, reshaped_encoded_sequences, sequence_labels)
# predict_and_graph_AS(LSTM_model, reshaped_encoded_sequences, anomaly_scores, contracted_anomalous_labels)
