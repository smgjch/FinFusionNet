import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.input_window_size = configs.seq_len
        self.kernel_size = configs.kernel_size
        self.num_filters = configs.num_kernels
        self.pred_len = configs.pred_len


        self.conv1_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                      kernel_size = self.kernel_size, dilation=1,padding =0)
        
        self.conv2_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                      kernel_size = self.kernel_size, dilation=2,padding = 0)
        
        self.conv3_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                      kernel_size = self.kernel_size, dilation=4,padding = 0)

        self.dense1 = nn.Linear(self.num_filters * self.input_window_size * 3, 4096 * 4)
        self.dense2 = nn.Linear(4096 * 4, 1024 * 4)
        self.dense3 = nn.Linear(1024 * 4, 1024 * 4)
        self.dense4 = nn.Linear(1024 * 4, 64)
        self.output_layer = nn.Linear(64, self.pred_len)

    def process_feature(self, input_feature):
        conv1 = F.relu(self.conv1_layers(input_feature))
        conv2 = F.relu(self.conv2_layers(input_feature))
        conv3 = F.relu(self.conv3_layers(input_feature))
        print(f"shapes {conv1.shape} {conv2.shape} {conv3.shape}")
        concatenated = torch.cat((conv1, conv2, conv3), dim=1)

        return concatenated

    def forward(self, inputs,_x,y,_y):
        # Split the input into separate features
        features = [inputs[:, i:i+1, :] for i in range(6)]

        # Process each feature separately
        processed_features = [self.process_feature(feature) for feature in features]

        # Concatenate the processed features along the feature dimension
        concatenated_features = torch.cat(processed_features, dim=2)

        # Flatten the concatenated features
        flattened_features = concatenated_features.view(concatenated_features.size(0), -1)

        # Dense layers (MLP)
        x = F.relu(self.dense1(flattened_features))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))

        # Output layer
        output = self.output_layer(x)

        return output
