import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Reshape, Concatenate, Flatten, Dense, Dropout
from tensorflow.keras import Model



class FFN_withDilation(Model):
    def __init__(self, num_filters=32, kernel_size=2,input_window_size=96):
        super(FFN_withDilation, self).__init__()
        self.input_window_size = input_window_size
        # self.kernel_size = kernel_size

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

        context_length = self.input_window_size
        conv1_reshaped = Reshape((context_length, self.num_filters))(conv1)
        conv2_reshaped = Reshape((context_length, self.num_filters))(conv2)
        conv3_reshaped = Reshape((context_length, self.num_filters))(conv3)

        concatenated = Concatenate(axis=2)([conv1_reshaped, conv2_reshaped, conv3_reshaped])

        return concatenated

    @tf.function
    def call(self, inputs, training=False):
        # print(f"inputs {inputs.shape}")
        # Split the input into separate features
        features = [Reshape((self.input_window_size, 1))(inputs[:, :, i]) for i in range(6)]

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



class FFN_without_dilation(Model):
    def __init__(self, num_filters=64, kernel_size=2,input_window_size=96):
        super(FFN_without_dilation, self).__init__()
        self.input_window_size = input_window_size
        # self.kernel_size = kernel_size

        self.conv1_layers = Conv1D(num_filters, kernel_size, dilation_rate=1, activation='relu', padding='same')
        # self.conv2_layers = Conv1D(num_filters, kernel_size, dilation_rate=2, activation='relu', padding='same')
        # self.conv3_layers = Conv1D(num_filters, kernel_size, dilation_rate=3, activation='relu', padding='same')
        initial_size = 24330
        self.dense1 = Dense(initial_size, activation='relu')
        # self.dropout1 = Dropout(0.2)
        self.dense2 = Dense(initial_size//2, activation='relu')
        # self.dropout2 = Dropout(0.2)
        self.dense3 = Dense(initial_size//4, activation='relu')
        self.dense4 = Dense(initial_size//8, activation='relu')
        self.dense5 = Dense(initial_size//16, activation='relu')
        self.dense6 = Dense(initial_size/32, activation='relu')
        self.dense7 = Dense(initial_size//64, activation='relu')
        # self.dropout3 = Dropout(0.2)
        self.output_layer = Dense(1, activation='linear')

    def process_feature(self, input_feature):
        conv1 = self.conv1_layers(input_feature)
        # conv2 = self.conv2_layers(input_feature)
        # conv3 = self.conv3_layers(input_feature)

        context_length = self.input_window_size
        conv1_reshaped = Reshape((context_length, self.num_filters))(conv1)
        # conv2_reshaped = Reshape((context_length, self.num_filters))(conv2)
        # conv3_reshaped = Reshape((context_length, self.num_filters))(conv3)

        concatenated = Concatenate(axis=2)([conv1_reshaped])

        return concatenated

    @tf.function
    def call(self, inputs, training=False):
        # print(f"inputs {inputs.shape}")
        # Split the input into separate features
        features = [Reshape((self.input_window_size , 1))(inputs[:, :, i]) for i in range(6)]

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
        x = self.dense5(x)
        x = self.dense6(x)
        x = self.dense7(x)
        # x = self.dropout3(x, training=training)

        # Output layer
        output = self.output_layer(x)

        return output


class FFN_withStep(Model):
    def __init__(self, num_filters=32, kernel_size=2,input_window_size=96):
        super(FFN_withStep, self).__init__()
        self.input_window_size = input_window_size
        # self.kernel_size = kernel_size

        self.conv1_layers = [Conv1D(num_filters, kernel_size, strides=1, activation='relu', padding='same') for _ in range(6)]
        self.conv2_layers = [Conv1D(num_filters, kernel_size, strides=2, activation='relu', padding='same') for _ in range(6)]
        self.conv3_layers = [Conv1D(num_filters, kernel_size, strides=3, activation='relu', padding='same') for _ in range(6)]

        self.flatten = Flatten()
        self.concat = Concatenate()

        self.dense1 = Dense(4096 * 4, activation='relu')
        self.dense2 = Dense(1024 * 4, activation='relu')
        self.dense3 = Dense(1024 * 4, activation='relu')
        self.dense4 = Dense(64, activation='relu')
        self.output_layer = Dense(1, activation='linear')

    def call(self, inputs):
        # Assuming inputs shape is (None, 32, 6)
        conv_outputs = []
        for i in range(6):
            x = inputs[:, :, i:i+1]  # Extract each feature
            x = self.conv1_layers[i](x)
            x = self.conv2_layers[i](x)
            x = self.conv3_layers[i](x)
            conv_outputs.append(self.flatten(x))

        concatenated = self.concat(conv_outputs)

        x = self.dense1(concatenated)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return self.output_layer(x)