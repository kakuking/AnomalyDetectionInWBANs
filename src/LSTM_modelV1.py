import tensorflow as tf
from keras.layers import LSTM, Embedding, Dense
from keras.models import Sequential
import numpy as np

'''
def createModel():
    # Define the model architecture
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))
    model.add(LSTM(64))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')
'''

# Load the database and correlate it
def load_and_correlate_dataset(INDEX_TO_CHECK)-> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    correlation_matrix: np.ndarray = np.load(base_path + "039_correlation.npy")
    encoded_data:       np.ndarray = np.load(base_path + "039_encoded_dataset.npy")
    labels:             np.ndarray = np.load(base_path + "039_LSTM_labels.npy")
    labels = labels.reshape(-1, 1)

    correlated_data = np.sum(combined_data * correlation_matrix[INDEX_TO_CHECK].T,
                             axis=1,
                             keepdims=True)

    print(f"Correlation matrix shape:       {correlation_matrix.shape}")
    print(f"Dataset shape:                  {combined_data.shape}")
    print(f"Labels shape:                   {labels.shape}")
    print(f"Correlated data shape:          {correlated_data.shape}")

    return correlation_matrix, combined_data, correlated_data, labels

# Find the anomaly scores
def find_anomaly_scores(correlated_data):
    correlated_mean = np.mean(correlated_data)
    correlated_std = np.std(correlated_data)
    anomaly_scores = (correlated_data - correlated_mean)/correlated_std

    print(f"Correlated data mean:           {correlated_mean}")
    print(f"Correlated data std:            {correlated_std}")
    print(f"Anomaly scores shape:           {anomaly_scores.shape}")

    return anomaly_scores

# Find anomalous_indices
def find_contextual_anomalies(anomaly_scores, labels):
    anomaly_std = np.std(anomaly_scores)
    indices_with_label = np.where(labels == 1)[0]

    anomalous_indices = np.where(anomaly_scores > anomaly_std)[0]

    print(f"Anomalous indices before filter:                {anomalous_indices.shape}")

    anomalous_indices = np.intersect1d(anomalous_indices, indices_with_label)

    print(f"Anomalous indices filtered by point anomaly:    {anomalous_indices.shape}")

    return anomalous_indices

def create_anomalous_labels():
    LSTM_labels = np.zeros(combined_data.shape[0])
    LSTM_labels[anomalous_indices] = 1
    
    return LSTM_labels

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

correlation_matrix, combined_data, correlated_data, labels = load_and_correlate_dataset(INDEX_TO_CHECK)
anomaly_scores = find_anomaly_scores(correlated_data)
anomalous_indices = find_contextual_anomalies(anomaly_scores, labels)
LSTM_labels = create_anomalous_labels()
