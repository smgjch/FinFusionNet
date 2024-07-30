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

        # self.conv1_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
        #                               kernel_size = self.kernel_size, dilation=1,padding =0)
        
        # self.conv1_1_layers = nn.Conv1d(in_channels = 1, out_channels = 1, 
        #                               kernel_size = self.kernel_size, dilation=1,padding =0,stride=2)
        
        # self.conv2_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
        #                               kernel_size = self.kernel_size, dilation=2,padding =0)
        # self.conv2_1_layers = nn.Conv1d(in_channels = 1, out_channels = 1, 
        #                               kernel_size = self.kernel_size, dilation=1,padding =0,stride=2)
        
        # self.conv3_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
        #                               kernel_size = self.kernel_size, dilation=3,padding =0)
        # self.conv3_1_layers = nn.Conv1d(in_channels = 1, out_channels = 1, 
        #                               kernel_size = self.kernel_size, dilation=1,padding =0,stride=2)
        

        input_size = self.enc_in*self.input_window_size
        input_size = int((input_size*3-1-2-3)*self.num_filters/2)
        attention_size = input_size//4+2

        self.input_layer = nn.Linear(self.enc_in*self.input_window_size,input_size)

        self.dense1 = nn.Linear(input_size, attention_size)
        # print(f"projection size {input_size//4+1}")
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=attention_size, nhead=3, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.output_layer = nn.Linear(attention_size,self.label_len)




    def forward(self, inputs,_x,y,_y):

        flattened_input = inputs.view(self.batch_size,1,-1)

        # convoluted_d1 = self.conv1_layers(flattened_input).view(self.batch_size,1,-1)
        # # print(f"shape of convoluted1 before stride x {convoluted_d1.shape}")

        # convoluted_d1 = self.conv1_1_layers(convoluted_d1)
        # # print(f"shape of convoluted1 x {convoluted_d1.shape}")

        # convoluted_d2 = self.conv2_layers(flattened_input).view(self.batch_size,1,-1)
        # # print(f"shape of convoluted2 before stride x {convoluted_d2.shape}")

        # convoluted_d2 = self.conv2_1_layers(convoluted_d2)
        # # print(f"shape of convoluted2 x {convoluted_d2.shape}")

        # convoluted_d3 = self.conv3_layers(flattened_input).view(self.batch_size,1,-1)
        # # print(f"shape of convoluted3 before stride x {convoluted_d3.shape}")

        # convoluted_d3 = self.conv3_1_layers(convoluted_d3)
        # # print(f"shape of convoluted3 x {convoluted_d3.shape}")
        
        # convoluted = torch.cat([convoluted_d1,convoluted_d2,convoluted_d3],dim=2)
        # # print(f"shape of convoluted x {convoluted.shape}")
        # convoluted = convoluted.view(self.batch_size, -1)  # Flatten for dense layers
        x = F.relu(self.input_layer(flattened_input))

        x = F.relu(self.dense1(x))
        
        x = self.transformer_encoder(x)

     
        output = self.output_layer(x).view(self.batch_size, 1, -1)  # Flatten for dense layers


        
        return output
    


class FlashAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, causal=False):
        print(f"shape of X {x.shape}")
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        x = flash_attn_func(q, k, v, causal=causal)
        x = x.reshape(B, N, C)
        x = self.out_proj(x)
        return x

class FlashTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = FlashAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src,src_mask,is_causal,src_key_padding_mask):
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src