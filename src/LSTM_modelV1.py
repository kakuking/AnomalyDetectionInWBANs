import tensorflow as tf
from keras.layers import LSTM, Embedding, Dense
from keras.models import Sequential
import numpy as np

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
    anomaly_scores = (correlated_data - correlated_mean)/correlated_std

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

    print(f"Anomalous indices after contraction from 7x7 to 7x1: {len(contracted_labels)}")

    return contracted_labels

base_path = "../numpy_saved_data/"
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
