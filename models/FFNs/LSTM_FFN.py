import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.AbsMaxPooling import MaxAbsolutePooling1D
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
        self.dropout_conv1 = nn.Dropout(p=0.5)  # Dropout after first conv layer

        self.pool1d = nn.MaxPool1d(kernel_size=2)


        # Set print options 
        torch.set_printoptions(threshold=100, edgeitems=5)

        input_size = 20*self.num_filters*self.enc_in//2
        # input_size = 4139
        LSTM_layers = 4
        hidden_size = 128

        self.projection1 = nn.Linear(input_size, input_size//2)
        self.dropout_P = nn.Dropout(p=0.5)  # Dropout after first conv layer

        self.lstm = nn.LSTM(input_size=input_size//2, 
                            hidden_size=hidden_size, 
                            num_layers=LSTM_layers, 
                            batch_first=True,
                            dropout=0.5)

        self.output_layer = nn.Linear(hidden_size, self.label_len)

    def patch_conv(self, inputs, window_size=20, stride=1):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        conved = torch.tensor([]).to(device)
        n, x, y = inputs.shape
        for i in range(0, x - window_size, stride):
            slice_ = inputs[:, i:i + window_size, :]
            # print(f"shape of current slice {slice_.shape}")

            sampled = self.t_sampling(slice_)
            # print(f"shape of convoulted slice {sampled.shape}")
            sampled = self.projection1(sampled)
            sampled = self.dropout_P(sampled)

            conved = torch.cat([conved,sampled],dim=1)
        return conved
    
    def t_sampling(self,inputs):
        flattened_input = inputs.view(self.batch_size, 1, -1)
        convoluted = self.conv1_layers(flattened_input)
        convoluted = self.dropout_conv1(convoluted)
        convoluted = convoluted.view(self.batch_size,1,-1)
        convoluted = self.pool1d(convoluted)

        return convoluted
    
    def forward(self, inputs, _x, y, _y):
        convoluted = self.patch_conv(inputs)
        # print(f"shape of before lstm {convoluted.shape}")
        output = self.lstm(convoluted)[1][0]

        output = output.permute(1,0,2)[:,-1:,:]
        # print(f"shape of lstm {output.shape}")

        output = self.output_layer(output)
        # print(f"shape of output {output.shape}")

        return output