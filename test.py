import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Reshape, Concatenate, Flatten, Dense, Dropout
from tensorflow.keras import Model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping,TensorBoard
from sklearn.preprocessing import MinMaxScaler


class FFN(Model):
    def __init__(self, num_filters=32, kernel_size=2):
        super(FFN, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        self.conv1_layers = Conv1D(num_filters, kernel_size, dilation_rate=1, activation='relu', padding='same')
        self.conv2_layers = Conv1D(num_filters, kernel_size, dilation_rate=2, activation='relu', padding='same')
        self.conv3_layers = Conv1D(num_filters, kernel_size, dilation_rate=3, activation='relu', padding='same')

        self.dense1 = Dense(4096 * 4, activation='relu')
        # self.dropout1 = Dropout(0.2)
        self.dense2 = Dense(1024 * 4, activation='relu')
        # self.dropout2 = Dropout(0.2)
        self.dense3 = Dense(1024 * 4, activation='relu')
        self.dense4 = Dense(64, activation='relu')
        # self.dropout3 = Dropout(0.2)
        self.output_layer = Dense(1, activation='linear')

    def process_feature(self, input_feature):
        conv1 = self.conv1_layers(input_feature)
        conv2 = self.conv2_layers(input_feature)
        conv3 = self.conv3_layers(input_feature)

        context_length = 96
        conv1_reshaped = Reshape((context_length, self.num_filters))(conv1)
        conv2_reshaped = Reshape((context_length, self.num_filters))(conv2)
        conv3_reshaped = Reshape((context_length, self.num_filters))(conv3)

        concatenated = Concatenate(axis=2)([conv1_reshaped, conv2_reshaped, conv3_reshaped])

        return concatenated

    @tf.function
    def call(self, inputs, training=False):
        # print(f"inputs {inputs.shape}")
        # Split the input into separate features
        features = [Reshape((96, 1))(inputs[:, :, i]) for i in range(6)]

        # print(f"features {features}")

        # Process each feature separately
        processed_features = [self.process_feature(feature) for feature in features]

        # Concatenate the processed features along the time dimension
        concatenated_features = Concatenate(axis=1)(processed_features)

        # Flatten the concatenated features
        flattened_features = Flatten()(concatenated_features)

        # Dense layers (MLP)
        x = self.dense1(flattened_features)
        # x = self.dropout1(x, training=training)
        x = self.dense2(x)
        # x = self.dropout2(x, training=training)
        x = self.dense3(x)
        x = self.dense4(x)
        # x = self.dropout3(x, training=training)

        # Output layer
        output = self.output_layer(x)

        return output


# data = pd.read_csv('data/ETTh2_head.csv')
data = pd.read_csv('data/ETTh2.csv')
target_col = 'OT'
X = data[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']].values
Y = data[target_col].values

scaler = MinMaxScaler()
X = scaler.fit_transform(X)


def create_sliding_windows(data, target, input_window_size, output_window_size):
    X = []
    Y = []
    for i in range(len(data) - input_window_size - output_window_size + 1):
        X.append(data[i:i + input_window_size])
        Y.append(target[i + input_window_size:i + input_window_size + output_window_size])
    return np.array(X), np.array(Y)


# Create sliding windows
input_window_size = 96
output_window_size = 1

x, y = create_sliding_windows(X, Y, input_window_size, output_window_size)

X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  train_size=0.8,
                                                  random_state=1)
#
#
# X_train, Y_train = create_sliding_windows(X_train, y_train, input_window_size, output_window_size)
#
# X_val, y_val = create_sliding_windows(X_val, y_val, input_window_size, output_window_size)
#
# X_test, y_test = create_sliding_windows(X_test, y_test, input_window_size, output_window_size)

print(x.shape)  # (num_samples, input_window_size, num_features)
print(y.shape)  # (num_samples, output_window_size)

# Define the input shape
input_shape = (None, input_window_size, X_train.shape[2])

# Create an instance of the model
model = FFN()

# Build the model by passing a dummy input through it
model.build(input_shape)

# Print the model summary


# Compile the model
model.compile(optimizer='adam', loss='mse')
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=15)
tensorboard_callback = TensorBoard(log_dir="logs")

history = model.fit(x=X_train,
                    y=y_train,
                    batch_size=64,
                    callbacks=[early_stopping,tensorboard_callback],
                    epochs=10000,
                    validation_data=[X_val, y_val])
