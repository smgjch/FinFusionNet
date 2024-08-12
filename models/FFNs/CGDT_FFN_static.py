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

        input_size = self.input_window_size
        attention_size = self.enc_in*self.input_window_size//6

        self.h_conv1 = GCNConv(in_channels=input_size, out_channels=input_size//4*3)
        self.h_conv2 = GCNConv(in_channels=input_size//4*3, out_channels=input_size//2) 
        self.h_conv3 = GCNConv(in_channels=input_size//2, out_channels=input_size//4) 
        self.h_conv4 = GCNConv(in_channels=input_size//4, out_channels=input_size//6) 
        # print(f"attention_size {attention_size}")
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=attention_size, nhead=3, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.output_layer = nn.Linear(attention_size,self.label_len)

    def forward(self, inputs,edge_index, edge_attr):

        inputs = inputs.permute(0,2,1)
        x = self.h_conv1(inputs, edge_index,edge_attr)

        x = F.relu(x)
        x = self.h_conv2(x, edge_index, edge_attr)
        x = F.relu(x)    

        x = self.h_conv3(x, edge_index, edge_attr)
        x = F.relu(x)        
        
        x = self.h_conv4(x, edge_index, edge_attr)
        x = F.relu(x)
        # print(f"shape of convoluted {x.shape}")
        x = x.view(self.batch_size,-1)
        # x = F.relu(self.projection(x))
        # print(f"attention size {x.shape}")
        # x = torch.mean(x, dim=0) 
        x = self.transformer_encoder(x)
        # if torch.isnan(x).any():
        #     print(f"transformer nan {x}")
        # print(f"shape of transfomered {x.shape}")
        output = self.output_layer(x).view(self.batch_size, 1, -1)  # Flatten for dense layers    
        # print(f"shape of output {output.shape}")
        # print(f"output {output}")
        # pred = output.detach().cpu()
        # input_test = inputs.detach().cpu()
        # if np.isnan(input_test).any():
        #     print(f"why input contains nan {input_test}")

        # if np.isnan(pred).any():
            
        #     print(f"------output is nan--------")
        #     print(f"{pred}")
        #     print(f"------inputs -------- \n {inputs}")
        #     print(f"------edge_index--------\n {edge_index}")
        #     print(f"------edge_attr--------\n {edge_attr}")



        
        return output
  