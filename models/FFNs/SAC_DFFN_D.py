#Stride ac deliation FFN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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
        




        input_size = self.enc_in*self.input_window_size
        input_size = int((input_size*3-1-2-3)*self.num_filters/2)

        self.dense1 = nn.Linear(input_size, input_size//4)
        self.dense2 = nn.Linear(input_size//4, input_size//8)
        self.dense3 = nn.Linear(input_size//8, input_size//16)
        self.dense4 = nn.Linear(input_size//16, input_size//32)
        self.dense5 = nn.Linear(input_size//32, input_size//64)
        self.dense6 = nn.Linear(input_size//64, input_size//128)
        self.dense7 = nn.Linear(input_size//128, 64)


        self.bn1 = nn.BatchNorm1d(input_size//4)
        self.bn2 = nn.BatchNorm1d(input_size // 8)



        self.output_layer = nn.Linear(64, self.label_len)
        # self.output_layerf = nn.Linear(16, 1)

        self.dropout = nn.Dropout(self.dropout_rate)


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

        # x = checkpoint(lambda x: F.dropout(F.relu(self.bn1(self.dense1(x))), p=self.dropout_rate_hidden, training=self.training), convoluted)
        # x = checkpoint(lambda x: F.dropout(F.relu(self.bn2(self.dense2(x))), p=self.dropout_rate_hidden, training=self.training), x)
        # x = checkpoint(lambda x: F.dropout(F.relu(self.bn3(self.dense3(x))), p=self.dropout_rate_hidden, training=self.training), x)
        # x = checkpoint(lambda x: F.relu(self.bn4(self.dense4(x))), x) 

        # print(f"shape of x {convoluted.shape}")

        x = checkpoint(lambda x: self.dropout(F.relu(self.bn1(self.dense1(x)))), convoluted)
        x = checkpoint(lambda x: self.dropout(F.relu(self.bn2(self.dense2(x)))), x)
        # x = checkpoint(lambda x: self.dropout(F.relu(self.dense3(x))), x)
        # x = checkpoint(lambda x: self.dropout(F.relu(self.dense4(x))), x)
        # x = checkpoint(lambda x: self.dropout(F.relu(self.dense5(x))), x)
        # x = checkpoint(lambda x: self.dropout(F.relu(self.dense6(x))), x)
        # x = checkpoint(lambda x: self.dropout(F.relu(self.dense7(x))), x)
        x = checkpoint(lambda x: F.relu(self.dense3(x)), x)
        x = checkpoint(lambda x: F.relu(self.dense4(x)), x)
        x = checkpoint(lambda x: F.relu(self.dense5(x)), x)
        x = checkpoint(lambda x: F.relu(self.dense6(x)), x)
        x = checkpoint(lambda x: F.relu(self.dense7(x)), x)
        # x = F.relu(self.dense1(convoluted))
        # x = F.relu(self.dense2(x))
        # x = F.relu(self.dense3(x))
        # x = F.relu(self.dense4(x))
        # x = F.relu(self.dense5(x))
        # x = F.relu(self.dense6(x))
        # x = F.relu(self.dense7(x))
        # print(f"x to output {x.shape}")
        
        # Output layer
        output = self.output_layer(x).view(self.batch_size, 1, -1)  # Flatten for dense layers

        # output = output.permute(0,2,1)
        # print(f"output {output.shape}")
        # output = self.output_layerf(output)

        
        return output
