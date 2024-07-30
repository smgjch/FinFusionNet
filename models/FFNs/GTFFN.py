#G for group convolution

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func


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

        self.conv1_layers = nn.Conv1d(in_channels = self.enc_in , out_channels = self.enc_in *self.num_filters, 
                                      kernel_size = self.kernel_size, dilation=1,groups=self.enc_in)
        
        self.conv1_1_layers = nn.Conv1d(in_channels = self.enc_in , out_channels = self.enc_in, 
                                      kernel_size = self.kernel_size, dilation=1,groups=self.enc_in, stride=2)
        
        self.conv2_layers = nn.Conv1d(in_channels =self.enc_in , out_channels = self.enc_in*self.num_filters , 
                                      kernel_size = self.kernel_size, dilation=2,groups=self.enc_in)
        
        self.conv2_1_layers = nn.Conv1d(in_channels = self.enc_in , out_channels = self.enc_in , 
                                      kernel_size = self.kernel_size, dilation=1,groups=self.enc_in,stride=2)
        
        self.conv3_layers = nn.Conv1d(in_channels = self.enc_in , out_channels = self.enc_in*self.num_filters , 
                                      kernel_size = self.kernel_size, dilation=3,groups=self.enc_in)
        
        self.conv3_1_layers = nn.Conv1d(in_channels = self.enc_in , out_channels = self.enc_in , 
                                      kernel_size = self.kernel_size, dilation=1,groups=self.enc_in,stride=2)
        


        input_size = self.input_window_size*self.num_filters
        input_size = int(((input_size//2-1) + (input_size//2-2) + (input_size//2-3)) * 138)

        attention_size = input_size//4
        # print(f"projection size {attention_size}")

        self.projection = nn.Linear(input_size, attention_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=attention_size, nhead=3, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.output_layer = nn.Linear(attention_size,self.label_len)




    def forward(self, inputs,_x,y,_y):

        flattened_input = inputs.permute(0,2,1)
        # print(f"shape of flattened_input x {flattened_input.shape}")

        convoluted_d1 = self.conv1_layers(flattened_input).view(self.batch_size,self.enc_in,-1)
        # print(f"shape of convoluted1 before stride x {convoluted_d1.shape}")

        convoluted_d1 = self.conv1_1_layers(convoluted_d1)
        # print(f"shape of convoluted1 x {convoluted_d1.shape}")

        convoluted_d2 = self.conv2_layers(flattened_input).view(self.batch_size,self.enc_in,-1)
        # print(f"shape of convoluted2 before stride x {convoluted_d2.shape}")

        convoluted_d2 = self.conv2_1_layers(convoluted_d2)
        # print(f"shape of convoluted2 x {convoluted_d2.shape}")

        convoluted_d3 = self.conv3_layers(flattened_input).view(self.batch_size,self.enc_in,-1)
        # print(f"shape of convoluted3 before stride x {convoluted_d3.shape}")

        convoluted_d3 = self.conv3_1_layers(convoluted_d3)
        # print(f"shape of convoluted3 x {convoluted_d3.shape}")
        
        convoluted = torch.cat([convoluted_d1,convoluted_d2,convoluted_d3],dim=2)
        # print(f"shape of cated x {convoluted.shape}")
        convoluted = convoluted.view(self.batch_size, -1)  # Flatten for dense layers
        # print(f"shape of flattened x {convoluted.shape}")

        x = F.relu(self.projection(convoluted))
        
        x = self.transformer_encoder(x)

     
        output = self.output_layer(x).view(self.batch_size, 1, -1)  # Flatten for dense layers


        
        return output
    