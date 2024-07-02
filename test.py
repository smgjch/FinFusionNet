import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Reshape, Concatenate, Flatten
from tensorflow.keras.models import Model


class FFN(Model):
    def __init__(self, num_filters=8, kernel_size=1):
        super(FFN, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        self.conv1_layers = Conv1D(num_filters, kernel_size, dilation_rate=1, activation='relu')
        self.conv2_layers = Conv1D(num_filters, kernel_size, dilation_rate=2, activation='relu')
        self.conv3_layers = Conv1D(num_filters, kernel_size, dilation_rate=4, activation='relu')

        # Define dense layers for the MLP
        self.dense1 = Dense(128, activation='relu')
        self.dropout1 = Dropout(0.2)
        self.dense2 = Dense(64, activation='relu')
        self.dropout2 = Dropout(0.2)
        self.dense3 = Dense(32, activation='relu')
        self.dropout3 = Dropout(0.2)
        self.output_layer = Dense(1, activation='linear')

    def process_feature(self, input_feature):
        conv1 = self.conv1_layers(input_feature)
        conv2 = self.conv2_layers(input_feature)
        conv3 = self.conv3_layers(input_feature)

        # Assume static shapes if you know them, otherwise calculate as required and assert with `tf.ensure_shape`
        expected_shape = 24  # Replace with your actual expected length post convolution
        conv1_reshaped = Reshape((expected_shape, self.num_filters))(conv1)
        conv2_reshaped = Reshape((expected_shape, self.num_filters))(conv2)
        conv3_reshaped = Reshape((expected_shape, self.num_filters))(conv3)

        concatenated = Concatenate(axis=2)([conv1_reshaped, conv2_reshaped, conv3_reshaped])

        return concatenated

    @tf.function
    def call(self, inputs, training=False):
        features = [Reshape((24, 1))(inputs[:, :, i]) for i in range(6)]

        # Process each feature separately
        processed_features = [self.process_feature(feature) for feature in features]

        # Concatenate the processed features along the time dimension
        concatenated_features = Concatenate(axis=1)(processed_features)

        # Flatten the concatenated features
        flattened_features = Flatten()(concatenated_features)

        # Dense layers (MLP)
        x = self.dense1(flattened_features)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        x = self.dropout3(x, training=training)

        # Output layer
        output = self.output_layer(x)

        return output


# Example instantiation and use
model = FFN()
# To build the model, you can run it on a sample input to determine all internal shapes automatically
dummy_input = tf.random.normal([1, 24, 6])  # Adjust dimensions as per your actual input
model(dummy_input)  # This will build the model
