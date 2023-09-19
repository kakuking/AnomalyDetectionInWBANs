import tensorflow as tf
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D
from keras.models import Sequential, Model, load_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt


BATCH_SIZE = 32
# TIME_STEPS = 1000
FEATURES = 1
TEST_SIZE = 0.2
VAL_SIZE = 0.1
NUM_EPOCHS = 5

INPUT_SHAPE = (7, 7)

spo2_data =     np.load("numpy_saved_data/spo2_data.npy")
pulse_data =    np.load("numpy_saved_data/pulse_data.npy")
hr_data =       np.load("numpy_saved_data/hr_data.npy")
resp_data =     np.load("numpy_saved_data/resp_data.npy")
abp_sys_data =  np.load("numpy_saved_data/abp_sys_data.npy")
abp_dia_data =  np.load("numpy_saved_data/abp_dia_data.npy")
abp_mean_data = np.load("numpy_saved_data/abp_mean_data.npy")

print("Loaded data")

combined_data = np.stack((spo2_data, pulse_data, hr_data, resp_data, abp_sys_data, abp_dia_data, abp_mean_data), axis=-1)

scaler = MinMaxScaler()
combined_data = scaler.fit_transform(combined_data)

subarray_shape = (7, 7)

num_subarrays = combined_data.shape[0] // subarray_shape[1]

subarrays = []

# Split the data into subarrays of shape (7, 10)
for i in range(num_subarrays):
    start_index = i * subarray_shape[1]
    end_index = start_index + subarray_shape[1]
    subarray = combined_data[start_index:end_index, :]
    subarrays.append(subarray)

subarrays = np.array(subarrays)
subarrays = np.expand_dims(subarrays, axis=-1)

print(f"Combined Data, shape of subarray: {subarrays.shape}")

# 0.7 train 0.2 test 0.1 valx
X_train, X_temp = train_test_split(subarrays, test_size=1-(TEST_SIZE + VAL_SIZE), random_state=42)
X_test, X_val = train_test_split(X_temp, test_size=0.33) 
X_train = X_train
X_val = X_val
X_test = X_test

#.reshape((-1,1))

print("normalized data and split into train and test sets")

input_shape = (7, 7, 1)  

INITIAL_ENCODING_DIM = 64

#Model def
input_layer = Input(shape=input_shape)
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

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

print("created model")

autoencoder.fit(X_train, X_train, epochs=2, batch_size=256, shuffle=True, validation_data=(X_test, X_test))

autoencoder.summary()

autoencoder.save("models/autoencoderV1-2.h5")


print("Model Saved")    # To load: tf.keras.models.load_model(model_path)

# autoencoder = load_model("models/autoencoderV1.h5")

Y_val = autoencoder.predict(X_val)

Y_val = np.squeeze(Y_val, axis=-1)
X_val = np.squeeze(X_val, axis=-1)

temp = Y_val.shape[0] * Y_val.shape[1]

Y_val = Y_val.reshape((temp, 7))
X_val = X_val.reshape((temp, 7))

val_MSE = mean_squared_error(X_val, Y_val)
val_MAE = mean_absolute_error(X_val, Y_val)

print(f"Mean Squared Error (MSE) on Validation Set: {val_MSE:.4f}")
print(f"Mean Absolute Error (MAE) on Validation Set: {val_MAE:.4f}")

plt.plot(X_val[:, 0], label="x")
plt.plot(Y_val[:, 0], label="y")

plt.grid()
# plt.ylim(-50, 400)
plt.legend(loc="upper right")
plt.title('ABP')

plt.ioff()
plt.tight_layout()

plt.show()