#D for deliation, T for transfomer, M for m sampling

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
        

        self.horizental_projection = nn.Linear(self.enc_in, 1)

        self.conv3d = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(2, 1, 1))

        horizental_convoluted_size =  self.enc_in*self.input_window_size
        horizental_projected_size = horizental_convoluted_size//2
        self.horizental_projection_2 = nn.Linear(horizental_convoluted_size, horizental_projected_size)


        v_feature_input_size = self.enc_in*self.input_window_size
        v_feature_input_size = int((v_feature_input_size*3-1-2-3)*self.num_filters/2)
        v_feature_output_size = v_feature_input_size//5+2
        self.vertical_projection = nn.Linear(v_feature_input_size, v_feature_output_size)
        
        concated_size = v_feature_output_size + horizental_projected_size

        self.h_encoder_layer = nn.TransformerEncoderLayer(d_model=horizental_projected_size, nhead=3, dim_feedforward=512)
        self.h_transformer_encoder = nn.TransformerEncoder(self.h_encoder_layer, num_layers=4)

        self.v_encoder_layer = nn.TransformerEncoderLayer(d_model=v_feature_output_size, nhead=3, dim_feedforward=512)
        self.v_transformer_encoder = nn.TransformerEncoder(self.v_encoder_layer, num_layers=4)
        self.output_layer = nn.Linear(concated_size, self.label_len)

    def h_sampling(self,inputs):
        timestamp_len = self.batch_size * self.input_window_size

        mask = torch.eye(self.enc_in, device=inputs.device).bool().unsqueeze(0).expand(timestamp_len, self.enc_in, self.enc_in)
        # print(f"mask {mask.shape}")
        
        x_reshaped = inputs.reshape(timestamp_len, 1,self.enc_in).expand(-1, self.enc_in, -1)
        # print(f"x_reshaped {x_reshaped.shape}")
        masked_input = x_reshaped.masked_fill(mask, 0).reshape(timestamp_len* self.enc_in, self.enc_in)
        # unmasked_input = x_reshaped.masked_select(~mask).reshape(timestamp_len* self.enc_in, self.enc_in - 1)
        # print(f"masked_input {masked_input.shape}")
        
        projected = self.horizental_projection(masked_input).reshape(self.batch_size, 1, self.input_window_size, self.enc_in)
        inputs = inputs.reshape(self.batch_size, 1, self.input_window_size, self.enc_in)
        # print(f"projected {projected.shape}, inputs {inputs.shape}")

        result = torch.stack([inputs, projected], dim=2)
        # result = result.reshape(128, 2, 1, 30, 138)
        # print(f"result {result.shape}")

        convoluted = self.conv3d(result)
        # print(f"convoluted {convoluted.shape}")
        convoluted = self.conv3d(result).reshape(self.batch_size,self.input_window_size*self.enc_in)
        # print(f"convoluted reshape{convoluted.shape}")
        return convoluted

    def v_sampling(self,inputs):
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

        return convoluted
    
    def forward(self, inputs,_x,y,_y):
        # print(f"input shape {inputs.shape}")
        h_features = self.h_sampling(inputs)
        v_features = self.v_sampling(inputs)
        # print(f"shape of h_features {h_features.shape}, v_features  {v_features.shape}")
        
        h_features = F.relu(self.horizental_projection_2(h_features))
        v_features = F.relu(self.vertical_projection(v_features))


        # print(f"shape of v_features after projection {v_features.shape}")

        # print(f"cated x {x.shape}")

        h_features = self.h_transformer_encoder(h_features)
        v_features = self.v_transformer_encoder(v_features)
        x = torch.cat([h_features,v_features],dim=1)

        # print(f"transformer procceed x {x.shape}")
     
        output = self.output_layer(x)
        # print(f"output {output.shape}")
        output = output.view(self.batch_size, 1, -1)
        # print(f"output.viewed {output.shape}")

        return output
    
