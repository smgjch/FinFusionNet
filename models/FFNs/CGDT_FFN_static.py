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

        self.h_conv1 = GCNConv(in_channels=self.enc_in, out_channels=16)
        self.h_conv2 = GCNConv(in_channels=16, out_channels=64) 

        input_size = self.enc_in*self.input_window_size
        input_size = int((input_size*3-1-2-3)*self.num_filters/2)
        attention_size = input_size//4+2
        # print(f"attention size {attention_size}")
        self.projection = nn.Linear(input_size, attention_size)
        # print(f"projection size {input_size//4+1}")
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=attention_size, nhead=3, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.output_layer = nn.Linear(attention_size,self.label_len)

    def forward(self, inputs,edge_index, edge_attr):
        # print(f"edge shape {edge_index.shape} \n {edge_index}\n-----\nedge attr {edge_attr.shape} \n {edge_attr}")
        x = self.h_conv1(inputs, edge_index,edge_attr)
        x = F.relu(x)
        x = self.h_conv2(x, edge_index, edge_attr)
        x = F.relu(x)

        x = F.relu(self.projection(x))
        # x = torch.mean(x, dim=0) 
        x = self.transformer_encoder(x)

        output = self.output_layer(x).view(self.batch_size, 1, -1)  # Flatten for dense layers    
        return output
  