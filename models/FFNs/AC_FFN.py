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
        self.gradient_checkpoint = configs.gradient_checkpoint

        self.conv1_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                      kernel_size = self.kernel_size, dilation=1,padding =0)
        input_size = int((self.enc_in*self.input_window_size-1)*self.num_filters)
        self.dense1 = nn.Linear(input_size, input_size//4)
        self.dense2 = nn.Linear(input_size//4, input_size//8)
        self.dense3 = nn.Linear(input_size//8, input_size//16)
        self.dense4 = nn.Linear(input_size//16, 64)
        self.output_layer = nn.Linear(64, self.label_len)
        # self.output_layerf = nn.Linear(16, 1)

    def process_feature(self, input_feature):
        to_concate = []
        for table in range(self.batch_size):
            page = []
            for i in range(self.input_window_size):
                # print(f"now at {i}")
                feature = input_feature[table:table+1, i:i+1, :]
                
                # feature = feature.permute(0,2,1)
                # print(f"sliced shape {feature.shape}")
                conved = F.relu(self.conv1_layers(feature))
                # print(f"conved shape {conved.shape}")
                
                conved = conved.flatten()
                # print(f"flatten shape {conved.shape}")

                page.append(conved)
            
            concatenated_page = torch.cat(page,dim=0)
            concatenated_page = torch.unsqueeze(concatenated_page,0)
            # print(f"page to concate {concatenated_page.shape}")
            
            to_concate.append(concatenated_page)

        # print(f"shapes {conv1.shape} {conv2.shape} {conv3.shape}")
        concatenated = torch.cat(to_concate, dim=0)

        return concatenated




    def forward(self, inputs, _x, y, _y):
        flattened_input = inputs.view(self.batch_size, 1, -1)
        if self.gradient_checkpoint:
        # Apply checkpointing to the convolutional layer
            convoluted = checkpoint(self.conv1_layers, flattened_input)
            convoluted = convoluted.view(self.batch_size, 1, -1)

            # Apply checkpointing to the dense layers
            x = checkpoint(lambda x: F.relu(self.dense1(x)), convoluted)
            x = checkpoint(lambda x: F.relu(self.dense2(x)), x)
            x = checkpoint(lambda x: F.relu(self.dense3(x)), x)
            x = checkpoint(lambda x: F.relu(self.dense4(x)), x)
            
            # Output layer (typically not checkpointed as it's the last layer)
            output = self.output_layer(x)

            # for name, param in self.named_parameters():
            #     if param.grad is not None:
            #         # print(f"{name}, {param.grad}, {global_step}")
            #         writer.add_histogram(f'gradients/{name}', param.grad, global_step)
        else:
            convoluted = self.conv1_layers(flattened_input)
            convoluted = convoluted.view(self.batch_size,1,-1)

            x = F.relu(self.dense1(convoluted))
            x = F.relu(self.dense2(x))
            x = F.relu(self.dense3(x))
            x = F.relu(self.dense4(x))
            # print(f"x to output {x.shape}")
            
            # Output layer
            output = self.output_layer(x)

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

        self.conv1_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                      kernel_size = self.kernel_size, dilation=1,padding =0)
        
        input_size = int((self.enc_in*self.input_window_size-1)*self.num_filters)

        self.dense1 = nn.Linear(input_size, input_size//4)
        self.dense2 = nn.Linear(input_size//4, input_size//8)
        self.dense3 = nn.Linear(input_size//8, input_size//16)
        self.dense4 = nn.Linear(input_size//16, 64)
        self.output_layer = nn.Linear(64, self.label_len)
        # self.output_layerf = nn.Linear(16, 1)



    def forward(self, inputs,_x,y,_y):
        flattened_input = inputs.view(self.batch_size,1,-1)
        convoluted = self.conv1_layers(flattened_input)
        convoluted = convoluted.view(self.batch_size,1,-1)

        x = F.relu(self.dense1(convoluted))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))
        # print(f"x to output {x.shape}")
        
        # Output layer
        output = self.output_layer(x)
        # output = output.permute(0,2,1)
        # print(f"output {output.shape}")
        # output = self.output_layerf(output)

        
        return output

