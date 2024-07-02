import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D

# Load your CSV data
# Assuming 'data.csv' contains columns: Open, High, Low, Close, AskVolume, BidVolume
# Adjust the file path and column names as needed
data = pd.read_csv('data.csv')

# Prepare your data
# Assuming you want to forecast 'Close' column
target_col = 'Close'
X = data[['Open', 'High', 'Low', 'Close', 'AskVolume', 'BidVolume']].values

# Reshape input data for Conv1D
X = np.expand_dims(X, axis=-1)  # Shape should be (samples, features, 1)


# Define the model
def build_model(input_shape):
    inputs = Input(shape=input_shape)

    # 1D Convolutional layers with increasing dilation rates
    conv1 = Conv1D(filters=32, kernel_size=3, dilation_rate=1, activation='relu')(inputs)
    conv2 = Conv1D(filters=32, kernel_size=3, dilation_rate=2, activation='relu')(inputs)
    conv3 = Conv1D(filters=32, kernel_size=3, dilation_rate=4, activation='relu')(inputs)

    # Print the shape after each convolutional layer
    print("Shape after conv1:", conv1.shape)
    print("Shape after conv2:", conv2.shape)
    print("Shape after conv3:", conv3.shape)

    model = Model(inputs=inputs, outputs=[conv1, conv2, conv3])
    return model


# Build the model
model = build_model(input_shape=X.shape[1:])

# Print the model summary
model.summary()
