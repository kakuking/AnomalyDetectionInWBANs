import tensorflow as tf
import numpy as np
from keras.layers import ConvLSTM2D, Flatten, Dense, Reshape, Dropout
from keras.models import Sequential, Model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


BATCH_SIZE = 32
# TIME_STEPS = 1000
FEATURES = 1
TEST_SIZE = 0.2
VAL_SIZE = 0.1
NUM_EPOCHS = 5

spo2_data =     np.load("numpy_saved_data/spo2_data.npy")
pulse_data =    np.load("numpy_saved_data/pulse_data.npy")
hr_data =       np.load("numpy_saved_data/hr_data.npy")
resp_data =     np.load("numpy_saved_data/resp_data.npy")
abp_sys_data =  np.load("numpy_saved_data/abp_sys_data.npy")
abp_dia_data =  np.load("numpy_saved_data/abp_dia_data.npy")
abp_mean_data = np.load("numpy_saved_data/abp_mean_data.npy")

print("Loaded data")

combined_data = np.stack((spo2_data, pulse_data, hr_data, resp_data, abp_sys_data, abp_dia_data, abp_mean_data), axis=-1)

print("Combined Data")

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(combined_data)

X_train, X_temp = train_test_split(normalized_data, test_size=1-(TEST_SIZE + VAL_SIZE), random_state=42)
X_test, X_val = train_test_split(X_temp, test_size=0.33) 
X_train = X_train
X_val = X_val
X_test = X_test

#.reshape((-1,1))

print("normalized data and split into train and test sets")

input_shape = (combined_data.shape[1], 1)  

INITIAL_ENCODING_DIM = 64

# model definition
input_layer = tf.keras.layers.Input(shape=input_shape)

encoder = tf.keras.layers.Dense(128, activation='relu')(input_layer)
encoder = tf.keras.layers.Dense(64, activation='relu')(encoder)  # Decrease encoding dim
encoder = tf.keras.layers.Dense(32, activation='relu')(encoder)  # Decrease encoding dim
encoded = tf.keras.layers.Dense(INITIAL_ENCODING_DIM, activation='relu')(encoder)  # Initial encoding dim

decoder = tf.keras.layers.Dense(32, activation='relu')(encoded)
decoder = tf.keras.layers.Dense(64, activation='relu')(decoder)
decoder = tf.keras.layers.Dense(128, activation='relu')(decoder)
decoded = tf.keras.layers.Dense(1, activation='sigmoid')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoded)

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

print("created model")

autoencoder.fit(X_train, X_train, epochs=1, batch_size=256, shuffle=True, validation_data=(X_test, X_test))

autoencoder.summary()

autoencoder.save("models/autoencoderV1.h5")

print("Model Saved")    # To load: tf.keras.models.load_model(model_path)

Y_val = autoencoder.predict(X_val)
Y_val = np.squeeze(Y_val, axis=-1)

val_MSE = mean_squared_error(X_val, Y_val)
val_MAE = mean_absolute_error(X_val, Y_val)

print(f"Mean Squared Error (MSE) on Validation Set: {val_MSE:.4f}")
print(f"Mean Absolute Error (MAE) on Validation Set: {val_MAE:.4f}")

