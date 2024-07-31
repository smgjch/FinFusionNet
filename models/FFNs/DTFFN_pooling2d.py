#Stride ac deliation FFN

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

        self.conv1_layers = nn.Conv2d(in_channels=1, out_channels= 4, kernel_size=4, stride=1)
        self.conv2_layers = nn.Conv2d(in_channels=4, out_channels= 16, kernel_size=2, stride=1)
        self.conv3_layers = nn.Conv2d(in_channels=16, out_channels= 32, kernel_size=2, stride=1)

     
        self.pool2d = nn.MaxPool2d(kernel_size=2,stride=2)



        input_size = 1024
        attention_size = 1024-1

        self.dense1 = nn.Linear(input_size, attention_size)
        # print(f"projection size {input_size//4+1}")
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=attention_size, nhead=3, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.output_layer = nn.Linear(attention_size,self.label_len)




    def forward(self, inputs,_x,y,_y):

        flattened_input = inputs.view(self.batch_size,1,self.input_window_size,self.enc_in)

        convoluted = self.conv1_layers(flattened_input)
        # print(f"shape of convoluted1 before stride x {convoluted_d1.shape}")

        convoluted_d1 = self.pool2d(convoluted)
        # print(f"shape of convoluted1 x {convoluted_d1.shape}")

        convoluted_d2 = self.conv2_layers(convoluted_d1)
        # print(f"shape of convoluted2 before stride x {convoluted_d2.shape}")
        convoluted_d2 = self.pool2d(convoluted_d2)

        convoluted_d3 = self.conv3_layers(convoluted_d2)
        # print(f"shape of convoluted2 before stride x {convoluted_d2.shape}")
        convoluted_d3 = self.pool2d(convoluted_d3)

      
        convoluted = convoluted_d3.view(self.batch_size, -1)  # Flatten for dense layers

        x = F.relu(self.dense1(convoluted))
        
        x = self.transformer_encoder(x)

     
        output = self.output_layer(x).view(self.batch_size, 1, -1)  # Flatten for dense layers


        
        return output
    

class Model_arch(nn.Module):
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

        self.conv1_layers = nn.Conv2d(in_channels=1, out_channels= self.num_filters, kernel_size=3, stride=1)
        self.conv2_layers = nn.Conv2d(in_channels=self.num_filters, out_channels= 1, kernel_size=2, stride=1)

     
        self.pool2d = nn.MaxPool2d(kernel_size=2,stride=2)



        input_size = 871
        attention_size = input_size+2
        # input_size = self.enc_in*self.input_window_size
        # input_size = int((input_size*3-1-2-3)*self.num_filters/2)
        # attention_size = input_size//4+2
        # print(f"attention size {attention_size}")
        self.dense1 = nn.Linear(input_size, attention_size)
        # print(f"projection size {input_size//4+1}")
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=attention_size, nhead=3, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.output_layer = nn.Linear(attention_size,self.label_len)




    def forward(self, inputs,_x,y,_y):

        flattened_input = inputs.view(self.batch_size,1,self.input_window_size,self.enc_in)

        convoluted = self.conv1_layers(flattened_input)
        # print(f"shape of convoluted1 before stride x {convoluted_d1.shape}")

        convoluted_d1 = self.pool2d(convoluted)
        # print(f"shape of convoluted1 x {convoluted_d1.shape}")

        convoluted_d2 = self.conv2_layers(convoluted_d1)
        # print(f"shape of convoluted2 before stride x {convoluted_d2.shape}")

      
        convoluted = convoluted_d2.view(self.batch_size, -1)  # Flatten for dense layers

        x = F.relu(self.dense1(convoluted))
        
        x = self.transformer_encoder(x)

     
        output = self.output_layer(x).view(self.batch_size, 1, -1)  # Flatten for dense layers


        
        return output
    