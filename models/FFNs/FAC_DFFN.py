#Fusion ac deliation FFN

import torch
import torch.nn as nn
import torch.nn.functional as F

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

        self.conv1_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                      kernel_size = self.kernel_size, dilation=1,padding =0)    
            
        self.conv1_1_layers = nn.Conv1d(in_channels = self.num_filters, out_channels = self.num_filters//2,
                                      kernel_size = self.kernel_size, dilation=1,padding =0)
        
        self.conv1_2_layers = nn.Conv1d(in_channels = self.num_filters//2, out_channels = 1,
                                      kernel_size = self.kernel_size, dilation=1,padding =0)
        
        
        
        self.conv2_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                      kernel_size = self.kernel_size, dilation=2,padding =0)
        
        self.conv2_1_layers = nn.Conv1d(in_channels = self.num_filters, out_channels = self.num_filters//2,
                                      kernel_size = self.kernel_size, dilation=1,padding =0)
        
        self.conv2_2_layers = nn.Conv1d(in_channels = self.num_filters//2, out_channels = 1,
                                      kernel_size = self.kernel_size, dilation=1,padding =0)
        

        
        self.conv3_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                      kernel_size = self.kernel_size, dilation=3,padding =0)
        
        self.conv3_1_layers = nn.Conv1d(in_channels = self.num_filters, out_channels = self.num_filters//2,
                                      kernel_size = self.kernel_size, dilation=1,padding =0)
        
        self.conv3_2_layers = nn.Conv1d(in_channels = self.num_filters//2, out_channels = 1,
                                      kernel_size = self.kernel_size, dilation=1,padding =0)

        input_size = int((self.enc_in*self.input_window_size-1)*self.num_filters)
        self.dense1 = nn.Linear(input_size, input_size//4)
        self.dense2 = nn.Linear(input_size//4, input_size//8)
        self.dense3 = nn.Linear(input_size//8, input_size//16)
        self.dense4 = nn.Linear(input_size//16, 64)
        self.output_layer = nn.Linear(64, self.label_len)
        # self.output_layerf = nn.Linear(16, 1)



    def forward(self, inputs,_x,y,_y):
        # print(f"shape of x {inputs.shape}")

        flattened_input = inputs.view(self.batch_size,1,-1)
        convoluted_d1 = self.conv1_layers(flattened_input)
        convoluted_d1_1 = self.conv1_1_layers(convoluted_d1)
        convoluted_d1_2 = self.conv1_2_layers(convoluted_d1_1)

        convoluted_d2 = self.conv2_layers(flattened_input)
        convoluted_d2_1 = self.conv2_1_layers(convoluted_d2)
        convoluted_d2_2 = self.conv2_2_layers(convoluted_d2_1)
        
        convoluted_d3 = self.conv3_layers(flattened_input)
        convoluted_d3_1 = self.conv3_1_layers(convoluted_d3)
        convoluted_d3_2 = self.conv3_2_layers(convoluted_d3_1)
        
        convoluted = torch.cat([convoluted_d1_2,convoluted_d2_2,convoluted_d3_2],dim=2)
        

        x = F.relu(self.dense1(convoluted))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))
        # print(f"x to output {x.shape}")
        
        # Output layer
        output = self.output_layer(x)
        # output = output.permute(0,2,1)
        # print(f"output {output.shape}")
        # output = self.output_layerf(output)

        
        return output
