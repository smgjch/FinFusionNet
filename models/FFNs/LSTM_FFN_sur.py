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
                                      kernel_size = self.kernel_size, dilation=1,padding ="same")
        
        self.pool1d = nn.MaxPool1d(kernel_size=2)
        

        input_size = self.enc_in*self.num_filters
        LSTM_layers = 1
        hidden_size = 256

        # self.projection1 = nn.Linear(input_size, input_size//2)

        print(f"input size is set to {input_size}")
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=LSTM_layers, 
                            batch_first=True)

        self.output_layer = nn.Linear(hidden_size, self.label_len)
    
    def t_sampling(self,inputs):
        flattened_input = inputs.view(self.batch_size, 1, -1)
        # print(f"shape of flattened_input {flattened_input.shape}")

        convoluted = self.conv1_layers(flattened_input)
        # print(f"shape of convoluted {convoluted.shape}")

        convoluted = convoluted.view(self.batch_size,1,-1)
        convoluted =  self.pool1d(convoluted)
        # print(f"shape of pool1d {convoluted.shape}")

        convoluted = convoluted.view(self.batch_size,-1,self.enc_in*self.num_filters)

        return convoluted
    
    def forward(self, inputs, _x, y, _y):
        # convoluted = self.patch_conv(inputs)
        convoluted = self.t_sampling(inputs)
        # print(f"shape of convoluted {convoluted.shape}")
        output = self.lstm(convoluted)[1][0]
        output = output.permute(1,0,2)
        # print(f"shape of lstm {output.shape}")

        output = self.output_layer(output)[:,-1:,:]
        # print(f"shape of output {output.shape}")

        return output