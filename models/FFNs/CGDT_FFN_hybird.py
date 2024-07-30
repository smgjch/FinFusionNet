#convolution graph deliation transfomer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
from torch_geometric.nn import GCNConv
from scipy.stats import pearsonr

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
        

        self.h_conv1 = GCNConv(in_channels=self.enc_in, out_channels=16)
        self.h_conv2 = GCNConv(in_channels=16, out_channels=64) 

        input_size = self.enc_in*self.input_window_size
        input_size = int((input_size*3-1-2-3)*self.num_filters/2)
        attention_size = input_size//4+2
        # print(f"attention size {attention_size}")
        self.dense1 = nn.Linear(input_size, attention_size)
        # print(f"projection size {input_size//4+1}")
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
        
        convoluted = torch.cat([convoluted_d1,convoluted_d2,convoluted_d3],dim=2)
        # print(f"shape of convoluted x {convoluted.shape}")
        convoluted = convoluted.view(self.batch_size, -1)  # Flatten for dense layers

        x = F.relu(self.dense1(convoluted))
        
        x = self.transformer_encoder(x)

     
        output = self.output_layer(x).view(self.batch_size, 1, -1)  # Flatten for dense layers


        
        return output
    
    @staticmethod
    def compute_pearson_correlation(x):
        """
        Compute the Pearson correlation matrix for the given input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (n, h, w) where n is batch size, h is length, and w is number of features.
        Returns:
            List of correlation matrices (one per sample in the batch).
        """
        batch_size, seq_len, num_features = x.shape
        correlation_matrices = []
        
        for i in range(batch_size):
            sample = x[i].numpy()  # Shape: (h, w)
            corr_matrix = np.zeros((num_features, num_features))
            for j in range(num_features):
                for k in range(num_features):
                    if j != k:
                        corr_matrix[j, k] = pearsonr(sample[:, j], sample[:, k])[0]
            correlation_matrices.append(corr_matrix)
        
        return torch.tensor(correlation_matrices, dtype=torch.float)