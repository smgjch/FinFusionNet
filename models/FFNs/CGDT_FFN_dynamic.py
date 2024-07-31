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

        input_size = int(self.input_window_size)

        self.h_conv1 = GCNConv(in_channels=input_size, out_channels=int(self.input_window_size//3*2))
        self.h_conv2 = GCNConv(in_channels=int(self.input_window_size//3*2), out_channels=int(self.input_window_size//3)) 

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.enc_in*int(self.input_window_size//3*2), nhead=10, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.output_layer = nn.Linear(self.enc_in*10,self.label_len)

    def forward(self, inputs,edge_index, edge_attr):
        # print(f"edges shape{edge_index.shape, edge_attr.shape }")
        # print(f"edges content {edge_index, edge_attr }")
        # edge_index = edge_index.long()
        # edge_attr = edge_attr.float()
        inputs = inputs.permute(0,2,1)
        x = self.h_conv1(inputs, edge_index,edge_attr)
        x = F.relu(x)
        x = self.h_conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        # print(f"shape of convoluted {x.shape}")
        x = x.view(self.batch_size,-1)
        # print(f"shape of convoluted {x.shape}")
        # x = F.relu(self.projection(x))
        # x = torch.mean(x, dim=0) 
        x = self.transformer_encoder(x)
        # print(f"shape of transfomered {x.shape}")
        output = self.output_layer(x).view(self.batch_size, 1, -1)  # Flatten for dense layers    
        return output
  