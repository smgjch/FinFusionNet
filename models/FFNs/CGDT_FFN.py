#convolution graph deliation transfomer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GCNConv
from scipy.stats import pearsonr
import numpy as np
from torch_geometric.data import Data, Batch

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
        self.GNN_type = configs.GNN_type

        h_conv_outputsize = int(self.input_window_size//3)

        self.h_conv1 = GCNConv(in_channels=self.input_window_size, out_channels=int(self.input_window_size//3*2))
        self.h_conv2 = GCNConv(in_channels=int(self.input_window_size//3*2), out_channels=h_conv_outputsize) 


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
        
        input_size = self.enc_in*self.input_window_size
        input_size = int((input_size*3-1-2-3)*self.num_filters/2)
        attention_size = input_size//4 + h_conv_outputsize

        print(f"attention size {attention_size}")
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=attention_size, nhead=10, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.output_layer = nn.Linear(attention_size,self.label_len)

    def v_sampling(self,inputs):
        flattened_input = inputs.view(self.batch_size,1,-1)

        convoluted_d1 = self.conv1_layers(flattened_input).view(self.batch_size,1,-1)
        convoluted_d1 = self.conv1_1_layers(convoluted_d1)

        convoluted_d2 = self.conv2_layers(flattened_input).view(self.batch_size,1,-1)
        convoluted_d2 = self.conv2_1_layers(convoluted_d2)

        convoluted_d3 = self.conv3_layers(flattened_input).view(self.batch_size,1,-1)
        convoluted_d3 = self.conv3_1_layers(convoluted_d3)
        
        convoluted = torch.cat([convoluted_d1,convoluted_d2,convoluted_d3],dim=2)
        convoluted = convoluted.view(self.batch_size, -1)  # Flatten for dense layers

        x = F.relu(self.dense1(convoluted))

        return x
    
    def h_sampling(self,inputs,edge_index, edge_attr):
        inputs = inputs.permute(0,2,1)
        x = self.h_conv1(inputs, edge_index,edge_attr)
        x = F.relu(x)
        x = self.h_conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = x.view(self.batch_size,-1)
        return x

    def forward(self, inputs,edge_index, edge_attr):
        v_features = self.v_sampling(inputs)
        print(f"V shape {v_features.shape}")

        h_features = self.h_sampling(inputs,edge_index, edge_attr)
        print(f"h shape {h_features.shape}")

        x = torch.cat([v_features,h_features],dim=2)
        print(f"catted shape {x.shape}")

        x = self.transformer_encoder(x)

        output = self.output_layer(x).view(self.batch_size, 1, -1)  # Flatten for dense layers    



        
        return output
  

class Mode_dual(nn.Module):
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
        self.GNN_type = configs.GNN_type

        h_conv_outputsize = int(self.input_window_size//3)

        self.h_conv1 = GCNConv(in_channels=self.input_window_size, out_channels=int(self.input_window_size//3*2))
        self.h_conv2 = GCNConv(in_channels=int(self.input_window_size//3*2), out_channels=h_conv_outputsize) 


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
        
        input_size = self.enc_in*self.input_window_size
        input_size = int((input_size*3-1-2-3)*self.num_filters/2)
        attention_size = input_size//4 + h_conv_outputsize

        print(f"attention size {attention_size}")
        self.v_encoder_layer = nn.TransformerEncoderLayer(d_model=attention_size, nhead=10, dim_feedforward=512)
        self.v_transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)        
        
        self.h_encoder_layer = nn.TransformerEncoderLayer(d_model=attention_size, nhead=10, dim_feedforward=512)
        self.h_transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.output_layer = nn.Linear(attention_size,self.label_len)

    def v_sampling(self,inputs):
        flattened_input = inputs.view(self.batch_size,1,-1)

        convoluted_d1 = self.conv1_layers(flattened_input).view(self.batch_size,1,-1)
        convoluted_d1 = self.conv1_1_layers(convoluted_d1)

        convoluted_d2 = self.conv2_layers(flattened_input).view(self.batch_size,1,-1)
        convoluted_d2 = self.conv2_1_layers(convoluted_d2)

        convoluted_d3 = self.conv3_layers(flattened_input).view(self.batch_size,1,-1)
        convoluted_d3 = self.conv3_1_layers(convoluted_d3)
        
        convoluted = torch.cat([convoluted_d1,convoluted_d2,convoluted_d3],dim=2)
        convoluted = convoluted.view(self.batch_size, -1)  # Flatten for dense layers

        x = F.relu(self.dense1(convoluted))
        
        return x
    
    def h_sampling(self,inputs,edge_index, edge_attr):
        inputs = inputs.permute(0,2,1)
        x = self.h_conv1(inputs, edge_index,edge_attr)
        x = F.relu(x)
        x = self.h_conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = x.view(self.batch_size,-1)
        return x

    def forward(self, inputs,edge_index, edge_attr):
        v_features = self.v_sampling(inputs)
        print(f"V shape {v_features.shape}")

        h_features = self.h_sampling(inputs,edge_index, edge_attr)
        print(f"h shape {h_features.shape}")

        x = torch.cat([v_features,h_features],dim=2)
        print(f"catted shape {x.shape}")

        x = self.transformer_encoder(x)

        output = self.output_layer(x).view(self.batch_size, 1, -1)  # Flatten for dense layers    



        
        return output
  