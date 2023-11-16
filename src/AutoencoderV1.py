import tensorflow as tf
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt
import csv


'''
Autoencoder V2 took a 7x7 input and created a 2x2x64 encoded output
Attempting now: AutoencoderV3, which will attempt to create the encoding for a 7x1 input : wont that make it contextual? Will attempt after consulting prof
'''

# loads data, combines it, normalizes it, creates 7x7 subarrays of it,
# makes the subarrays global, hence no return
def load_data():
    base_path = "../numpy_saved_data/"
    spo2_data =     np.load(base_path + "spo2_data.npy")
    pulse_data =    np.load(base_path + "pulse_data.npy")
    hr_data =       np.load(base_path + "hr_data.npy")
    resp_data =     np.load(base_path + "resp_data.npy")
    abp_sys_data =  np.load(base_path + "abp_sys_data.npy")
    abp_dia_data =  np.load(base_path + "abp_dia_data.npy")
    abp_mean_data = np.load(base_path + "abp_mean_data.npy")

    print("Loaded data")

    combined_data = np.stack((spo2_data, pulse_data, hr_data, resp_data, abp_sys_data, abp_dia_data, abp_mean_data), axis=-1)

    contains_zero = np.any(combined_data == 0, axis=1)

    # Use boolean indexing to remove rows containing 0
    combined_data = combined_data[~contains_zero]
    print(f"Removed all rows containing 0: {combined_data.shape}")

    scaler = MinMaxScaler()
    combined_data = scaler.fit_transform(combined_data)

    subarray_shape = (7, 7)

    num_subarrays = combined_data.shape[0] // subarray_shape[1]

    global subarrays
    subarrays = []

    # Split the data into subarrays of shape (7, 10)
    for i in range(num_subarrays):
        start_index = i * subarray_shape[1]
        end_index = start_index + subarray_shape[1]
        subarray = combined_data[start_index:end_index, :]
        subarrays.append(subarray)

    subarrays = np.array(subarrays)
    subarrays = np.expand_dims(subarrays, axis=-1)

    # np.save(base_path + "039_combined_subarrays.npy",   subarrays)
    print(f"Combined/Normalized Data, shape of subarray: {subarrays.shape}")

# Splits previous data into train-test-validate set
def split_data() -> [[], [], []]:
    # 0.7 train 0.2 test 0.1 valx
    X_train, X_temp = train_test_split(subarrays, test_size=(TEST_SIZE + VAL_SIZE), random_state=42)
    X_test, X_val = train_test_split(X_temp, test_size=VAL_SIZE/(TEST_SIZE + VAL_SIZE)) 
    X_train = X_train
    X_val = X_val
    X_test = X_test
    #.reshape((-1,1))
    print("split data into train, test, and validate sets")

    return X_train, X_test, X_val

# Creates model
def create_model() -> [Model, EarlyStopping, ReduceLROnPlateau]:
    #Model def
    input_layer = Input(shape=INPUT_SHAPE)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Define the decoder architecture
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Reshape the output to match the input shape (7, 10)
    cropped = Cropping2D(cropping=((0, 1), (0, 1)))(decoded)

    autoencoder = Model(inputs=input_layer, outputs=cropped)

    autoencoder.compile(optimizer='adam', loss='mean_absolute_error')  # binary_crossentropy mean_absolute_error

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=int(1e-6))

    # print("created model")
    return autoencoder, early_stopping, reduce_lr

# Trains model m
def train_model(autoencoder, early_stopping, reduce_lr, model_path):
    autoencoder.fit(X_train, X_train,
                    epochs=NUM_EPOCHS,
                    batch_size=BATCH_SIZE,
                    shuffle=True, 
                    validation_data=(X_test, X_test),
                    callbacks=[early_stopping, reduce_lr]
                    )



    autoencoder.summary()

    autoencoder.save(model_path)

    print("Model Saved")    # To load: tf.keras.models.load_model(model_path)

    return autoencoder

# predicts the given value, calculates MSE, MAE, SD 
# uses SD to check if anomaly or not,
# creates a plot to show anomalies
# Index to check is which physiological data to show,
# How many anomalies is how many anomalies on teh graph
def predict_validate_metric_graph(X_val, INDEX_TO_CHECK, how_many_anomalies):
    Y_val = autoencoder.predict(X_val)

    Y_val = np.squeeze(Y_val, axis=-1)
    X_val = np.squeeze(X_val, axis=-1)

    temp = Y_val.shape[0] * Y_val.shape[1]

    Y_val = Y_val.reshape((temp, 7))
    X_val = X_val.reshape((temp, 7))

    abs_diff = np.abs(Y_val[:, INDEX_TO_CHECK] - X_val[:, INDEX_TO_CHECK])

    val_MSE = mean_squared_error(X_val[:, INDEX_TO_CHECK], Y_val[:, INDEX_TO_CHECK])
    val_MAE = mean_absolute_error(X_val[:, INDEX_TO_CHECK], Y_val[:, INDEX_TO_CHECK])
    error_SD = np.std(Y_val[:, INDEX_TO_CHECK] - X_val[:, INDEX_TO_CHECK])

    ANOMALIES = np.where(abs_diff > error_SD)[0]

    print(f"Number of anomalies: {ANOMALIES.shape[0]}")
    print(f"Mean Squared Error (MSE) on Validation Set: {val_MSE:.4f}")
    print(f"Mean Absolute Error (MAE) on Validation Set: {val_MAE:.4f}")
    print(f"Standard Deviation of Error on Validation Set: {error_SD:.4f}")

    ending_index = ANOMALIES[how_many_anomalies]
    if how_many_anomalies == -1:
        ending_index = -1
    
    plt.plot(X_val[: ending_index, INDEX_TO_CHECK], label="Actual Values")
    plt.plot(Y_val[: ending_index, INDEX_TO_CHECK], label="Predicted Values")
    plt.plot(ANOMALIES[:how_many_anomalies], X_val[ANOMALIES[:how_many_anomalies], INDEX_TO_CHECK], 'ro', label='Difference > Threshold')

    plt.grid()
    plt.title('For ' + INDEX_VS_METRIC[INDEX_TO_CHECK])
    plt.legend(loc="upper right")

    plt.ioff()
    plt.tight_layout()

    plt.show()

def predict_and_save(Input, INDEX_TO_CHECK) -> np.ndarray:
    Y_val = autoencoder.predict(Input)

    Y_val = np.squeeze(Y_val, axis=-1)
    Input = np.squeeze(Input, axis=-1)

    temp = Y_val.shape[0] * Y_val.shape[1]

    Y_val = Y_val.reshape((temp, 7))
    Input = Input.reshape((temp, 7))

    abs_diff = np.abs(Y_val[:, INDEX_TO_CHECK] - Input[:, INDEX_TO_CHECK])

    val_MSE = mean_squared_error(Input[:, INDEX_TO_CHECK], Y_val[:, INDEX_TO_CHECK])
    val_MAE = mean_absolute_error(Input[:, INDEX_TO_CHECK], Y_val[:, INDEX_TO_CHECK])
    error_SD = np.std(Y_val[:, INDEX_TO_CHECK] - Input[:, INDEX_TO_CHECK])

    ANOMALIES = np.where(abs_diff > error_SD)[0]

    labels = np.zeros(Y_val.shape[0], dtype=int)
    labels[ANOMALIES] = 1

    np.save(base_path + "039_LSTM_labels.npy",  labels)
    np.save(base_path + "039_combined_dataset.npy", Input)

    return labels

def create_encoder(autoencoder) -> Model:
    encoder_layers = autoencoder.layers[:5]
    encoder = Model(inputs=autoencoder.input, outputs=encoder_layers[-1].output)

    encoder.save("../models/encoderV2.h5")

    return encoder

def encode_and_save(Input, encoder):
    encoded_data = encoder.predict(Input)

    np.save(base_path + "039_encoded_dataset.npy", encoded_data)

def epochsVsAnomalies(autoencoder, early_stopping, reduce_lr, number_epochs, X_val, INDEX_TO_CHECK):
    autoencoder.fit(X_train, X_train,
                    epochs=number_epochs,
                    batch_size=BATCH_SIZE,
                    shuffle=True, 
                    validation_data=(X_test, X_test),
                    callbacks=[early_stopping, reduce_lr],
                    verbose=0
                    )

    Y_val = autoencoder.predict(X_val, verbose=0)

    Y_val = np.squeeze(Y_val, axis=-1)
    X_val = np.squeeze(X_val, axis=-1)

    temp = Y_val.shape[0] * Y_val.shape[1]

    Y_val = Y_val.reshape((temp, 7))
    X_val = X_val.reshape((temp, 7))

    abs_diff = np.abs(Y_val[:, INDEX_TO_CHECK] - X_val[:, INDEX_TO_CHECK])

    val_MSE = mean_squared_error(X_val[:, INDEX_TO_CHECK], Y_val[:, INDEX_TO_CHECK])
    val_MAE = mean_absolute_error(X_val[:, INDEX_TO_CHECK], Y_val[:, INDEX_TO_CHECK])
    error_SD = np.std(Y_val[:, INDEX_TO_CHECK] - X_val[:, INDEX_TO_CHECK])

    ANOMALIES = np.where(abs_diff > error_SD)[0]
    
    with open(csv_file_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([str(number_epochs),str(len(ANOMALIES)), str(val_MAE), str(val_MSE)])
    

base_path = "../numpy_saved_data/"
model_path = "../models/autoencoderV2.h5"
BATCH_SIZE = 128
TEST_SIZE = 0.2
VAL_SIZE = 0.1
NUM_EPOCHS = 20
csv_file_path = "./autoencoder.csv"
INDEX_VS_METRIC = ["sp02", "Pulse", "Heart Rate", "Respiration Rate", "ABP Systolic", "ABP Diasotlic", "ABP Mean"]

INPUT_SHAPE = (7, 7, 1)
INITIAL_ENCODING_DIM = 64

load_data()
X_train, X_test, X_val = split_data()
autoencoder, early_stopping, reduce_lr = create_model()

# autoencoder = train_model(autoencoder, early_stopping, reduce_lr, model_path)
autoencoder = load_model(model_path)

predict_validate_metric_graph(subarrays, 0, -1)    # -1 means show all anomalies
labels = predict_and_save(subarrays, 0)                                                     # CHANGE THIS ONE 

encoder = create_encoder(autoencoder)
encode_and_save(subarrays, encoder)

# spo2_data ========== 0
# pulse_data ========= 1
# hr_data ============ 2
# resp_data ========== 3
# abp_sys_data ======= 4
# abp_dia_data ======= 5
# abp_mean_data ====== 6
