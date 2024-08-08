#Stride ac deliation FFN

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    
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
        self.dropout_rate = configs.dropout

        self.conv1_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                      kernel_size = self.kernel_size, dilation=1,padding =0)
        self.conv1_1_layers = nn.Conv1d(in_channels = 1, out_channels = 1, 
                                      kernel_size = self.kernel_size, dilation=1,padding =0,stride=2)
        
        self.conv2_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                      kernel_size = self.kernel_size, dilation=2,padding =0)
        self.conv2_1_layers = nn.Conv1d(in_channels = 1, out_channels = 1, 
                                      kernel_size = self.kernel_size, dilation=1,padding =0,stride=2)
        
        self.conv3_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                      kernel_size = self.kernel_size, dilation=3,padding =0)
        self.conv3_1_layers = nn.Conv1d(in_channels = 1, out_channels = 1, 
                                      kernel_size = self.kernel_size, dilation=1,padding =0,stride=2)
        

        self.pos_encoder1 = PositionalEncoding(1342)
        self.pos_encoder2 = PositionalEncoding(1340)
        self.pos_encoder3 = PositionalEncoding(1338)


        self.layer_norm1 = nn.LayerNorm(1342)

        input_size = self.enc_in*self.input_window_size
        input_size = int((input_size*3-1-2-3)*self.num_filters/2)
        attention_size = input_size//4+2

        self.dense1 = nn.Linear(input_size, attention_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=attention_size, nhead=3, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        
        self.output_layer = nn.Linear(attention_size,self.label_len)


    def forward(self, inputs,_x,y,_y):

        flattened_input = inputs.view(self.batch_size,1,-1)

        convoluted_d1 = self.conv1_layers(flattened_input).view(self.batch_size,1,-1)
        # print(f"shape of convoluted1 before stride x {convoluted_d1.shape}")

        convoluted_d1 = self.conv1_1_layers(convoluted_d1)
        # print(f"shape of convoluted1 x {convoluted_d1.shape}")

        convoluted_d2 = self.conv2_layers(flattened_input).view(self.batch_size,1,-1)
        # print(f"shape of convoluted2 before stride x {convoluted_d2.shape}")

        convoluted_d2 = self.conv2_1_layers(convoluted_d2)
        # print(f"shape of convoluted2 x {convoluted_d2.shape}")

        convoluted_d3 = self.conv3_layers(flattened_input).view(self.batch_size,1,-1)
        # print(f"shape of convoluted3 before stride x {convoluted_d3.shape}")

        convoluted_d3 = self.conv3_1_layers(convoluted_d3)
        # print(f"shape of convoluted3 x {convoluted_d3.shape}")
        convoluted_d1 = self.pos_encoder1(convoluted_d1.transpose(0, 1)).transpose(0, 1)
        convoluted_d2 = self.pos_encoder2(convoluted_d2.transpose(0, 1)).transpose(0, 1)
        convoluted_d3 = self.pos_encoder3(convoluted_d3.transpose(0, 1)).transpose(0, 1)        
        
        convoluted = torch.cat([convoluted_d1,convoluted_d2,convoluted_d3],dim=2)
        # print(f"shape of convoluted x {convoluted.shape}")
        convoluted = convoluted.view(self.batch_size, -1)  # Flatten for dense layers
        convoluted = self.layer_norm1(convoluted)

        x = F.relu(self.dense1(convoluted))
        
        x = self.transformer_encoder(x)

        output = self.output_layer(x).view(self.batch_size, 1, -1)  # Flatten for dense layers
        
        return output
    