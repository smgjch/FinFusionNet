#Masked FFN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.verbose = configs.verbose
        self.input_window_size = configs.seq_len
        self.kernel_size = configs.kernel_size
        self.num_filters = configs.num_kernels
        self.pred_len = configs.label_len
        self.enc_in = configs.enc_in
        self.batch_size = configs.batch_size

        self.projection = nn.Linear(self.enc_in, 1)

        self.conv3d = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(2, 1, 1))

        input_size = int(self.enc_in*self.input_window_size)
        self.dense1 = nn.Linear(input_size, input_size//4)
        self.dense2 = nn.Linear(input_size//4, input_size//8)
        self.dense3 = nn.Linear(input_size//8, input_size//16)
        self.dense4 = nn.Linear(input_size//16, 64)
        self.output_layer = nn.Linear(64, self.pred_len)



    def forward(self, inputs,_x,y,_y):
        # print(f"input {inputs.shape}"ç)
        timestamp_len = self.batch_size * self.input_window_size

        mask = torch.eye(self.enc_in, device=inputs.device).bool().unsqueeze(0).expand(timestamp_len, self.enc_in, self.enc_in)
        # print(f"mask {mask.shape}")
        
        x_reshaped = inputs.reshape(timestamp_len, 1,self.enc_in).expand(-1, self.enc_in, -1)
        # print(f"x_reshaped {x_reshaped.shape}")
        masked_input = x_reshaped.masked_fill(mask, 0).reshape(timestamp_len* self.enc_in, self.enc_in)
        # unmasked_input = x_reshaped.masked_select(~mask).reshape(timestamp_len* self.enc_in, self.enc_in - 1)
        # print(f"masked_input {masked_input.shape}")
        
        projected = self.projection(masked_input).reshape(self.batch_size, 1, self.input_window_size, self.enc_in)
        inputs = inputs.reshape(self.batch_size, 1, self.input_window_size, self.enc_in)
        # print(f"projected {projected.shape}, inputs {inputs.shape}")

        result = torch.stack([inputs, projected], dim=2)
        # result = result.reshape(128, 2, 1, 30, 138)
        # print(f"result {result.shape}")

        convoluted = self.conv3d(result)
        # print(f"convoluted {convoluted.shape}")
        convoluted = self.conv3d(result).reshape(self.batch_size,self.input_window_size*self.enc_in)
        # print(f"convoluted reshape{convoluted.shape}")


        x = F.relu(self.dense1(convoluted))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))

        output = self.output_layer(x)
        # print(f"input {inputs.shape}")

        output = torch.unsqueeze(output,1)
        # print(f"input {inputs.shape}")

        return output
