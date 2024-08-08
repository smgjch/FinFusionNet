import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        self.conv2_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                    kernel_size = self.kernel_size, dilation=1,padding =0)
        
        self.conv3_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                    kernel_size = self.kernel_size, dilation=1,padding =0)
                
        self.conv4_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                    kernel_size = self.kernel_size, dilation=1,padding =0)
                
        self.conv5_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                    kernel_size = self.kernel_size, dilation=1,padding =0)
        
        self.conv12_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                     kernel_size = self.kernel_size, dilation=1,padding =0)
        
        self.conv22_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                    kernel_size = self.kernel_size, dilation=1,padding =0)
        
        self.conv32_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                    kernel_size = self.kernel_size, dilation=1,padding =0)
                
        self.conv42_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                    kernel_size = self.kernel_size, dilation=1,padding =0)
                
        self.conv52_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                    kernel_size = self.kernel_size, dilation=1,padding =0)
        
        self.pool1d = nn.MaxPool1d(kernel_size=2,stride = 2)

        input_size = int((self.enc_in*self.input_window_size-1)*self.num_filters)//2
        
        self.dense1 = nn.Linear(4130, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64,32)
        self.dense4 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, self.label_len)

    def forward(self, inputs, _x, y, _y):
        flattened_input = inputs.view(self.batch_size, 1, -1)

        convoluted = self.conv1_layers(flattened_input)
        convoluted =  self.pool1d(convoluted).view(self.batch_size, 1, -1)

        convoluted = self.conv2_layers(convoluted)
        convoluted =  self.pool1d(convoluted).view(self.batch_size, 1, -1)
        
        convoluted = self.conv3_layers(convoluted)
        convoluted =  self.pool1d(convoluted).view(self.batch_size, 1, -1)

        convoluted = self.conv4_layers(convoluted)
        convoluted =  self.pool1d(convoluted).view(self.batch_size, 1, -1)

        convoluted = self.conv5_layers(convoluted)
        convoluted =  self.pool1d(convoluted).view(self.batch_size, 1, -1)        
        
        convoluted = self.conv12_layers(flattened_input)
        convoluted =  self.pool1d(convoluted).view(self.batch_size, 1, -1)

        convoluted = self.conv22_layers(convoluted)
        convoluted =  self.pool1d(convoluted).view(self.batch_size, 1, -1)
        
        convoluted = self.conv32_layers(convoluted)
        convoluted =  self.pool1d(convoluted).view(self.batch_size, 1, -1)

        convoluted = self.conv42_layers(convoluted)
        convoluted =  self.pool1d(convoluted).view(self.batch_size, 1, -1)

        convoluted = self.conv52_layers(convoluted)
        convoluted =  self.pool1d(convoluted).view(self.batch_size, 1, -1)

        x = F.relu(self.dense1(convoluted))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))
 
        output = self.output_layer(x)

        return output

