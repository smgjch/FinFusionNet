import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class Model_arch(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.verbose = configs.verbose
        self.input_window_size = configs.seq_len
        self.kernel_size = configs.kernel_size
        self.num_filters = configs.num_kernels
        self.pred_len = configs.label_len

        self.conv1_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                      kernel_size = self.kernel_size, dilation=1,padding =0)
        
        self.conv2_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                      kernel_size = self.kernel_size, dilation=2,padding = 0)
        
        self.conv3_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                      kernel_size = self.kernel_size, dilation=4,padding = 0)

        self.dense1 = nn.Linear(2442 , 4096 * 4)
        self.dense2 = nn.Linear(4096 * 4, 1024 * 4)
        self.dense3 = nn.Linear(1024 * 4, 1024 * 4)
        self.dense4 = nn.Linear(1024 * 4, 64)
        self.output_layer = nn.Linear(64, self.pred_len)
        self.output_layerf = nn.Linear(16, 1)

    def process_feature(self, input_feature):
        conv1 = F.relu(self.conv1_layers(input_feature))
        conv2 = F.relu(self.conv2_layers(input_feature))
        conv3 = F.relu(self.conv3_layers(input_feature))
        # print(f"shapes {conv1.shape} {conv2.shape} {conv3.shape}")
        concatenated = torch.cat((conv1, conv2, conv3), dim=2)

        return concatenated

    def forward(self, inputs,_x,y,_y):
        # print(f"shape of input {inputs.shape}")
        # Split the input into separate features
        features = [inputs[:, i:i+1, :] for i in range(6)]
        # features = [inputs[:, :, i:i+1] for i in range(138)]

        # Process each feature separately
        processed_features = [self.process_feature(feature) for feature in features]

        # Concatenate the processed features along the feature dimension
        concatenated_features = torch.cat(processed_features, dim=2)
        # print(f"before flatten {concatenated_features.shape}")
        # Flatten the concatenated features
        # flattened_features = concatenated_features.view(concatenated_features.size(0), -1)
        # print(f"flattened {flattened_features.shape}")
        # Dense layers (MLP)
        x = F.relu(self.dense1(concatenated_features))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))

        # Output layer
        output = self.output_layer(x)
        output = output.permute(0,2,1)
        # print(f"output {output.shape}")
        output = self.output_layerf(output)
        # print(f"outputf {output.shape}")
        
        return output
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.verbose = configs.verbose
        self.input_window_size = configs.seq_len
        self.label_len = configs.label_len
        self.batch_size = configs.batch_size
        self.kernel_size = configs.kernel_size
        self.num_filters = configs.num_kernels
        self.enc_in = configs.enc_in
        self.gradient_checkpoint = configs.gradient_checkpoint


        self.conv1_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                      kernel_size = self.kernel_size, dilation=1,padding =0)
        
        self.conv2_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                      kernel_size = self.kernel_size, dilation=2,padding =0)
        
        self.conv3_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                      kernel_size = self.kernel_size, dilation=3,padding =0)
        
        unconvuluted_size = self.enc_in*self.input_window_size
        input_size = int((unconvuluted_size*3-1-2-3)*self.num_filters) # theree convulution layers, with increasing 
                                                                        #dilation, casue a shrink of lenght of 1,2,and 3
        self.dense1 = nn.Linear(input_size, input_size//4)
        self.dense2 = nn.Linear(input_size//4, input_size//8)
        self.dense3 = nn.Linear(input_size//8, input_size//16)
        self.dense4 = nn.Linear(input_size//16, 64)
        self.output_layer = nn.Linear(64, self.label_len)
        # self.output_layerf = nn.Linear(16, 1)



    def forward(self, inputs,_x,y,_y):
        # print(f"shape of x {inputs.shape}")

        flattened_input = inputs.view(self.batch_size,1,-1)
        convoluted_d1 = self.conv1_layers(flattened_input).view(self.batch_size,1,-1)
        convoluted_d2 = self.conv2_layers(flattened_input).view(self.batch_size,1,-1)
        convoluted_d3 = self.conv3_layers(flattened_input).view(self.batch_size,1,-1)
        
        convoluted = torch.cat([convoluted_d1,convoluted_d2,convoluted_d3],dim=2)
        
        if self.gradient_checkpoint:
            x = checkpoint(lambda x: F.relu(self.dense1(x)), convoluted)
            x = checkpoint(lambda x: F.relu(self.dense2(x)), x)
            x = checkpoint(lambda x: F.relu(self.dense3(x)), x)
            x = checkpoint(lambda x: F.relu(self.dense4(x)), x)

        else:
            x = F.relu(self.dense1(convoluted))
            x = F.relu(self.dense2(x))
            x = F.relu(self.dense3(x))
            x = F.relu(self.dense4(x))

        output = self.output_layer(x)


        
        return output
