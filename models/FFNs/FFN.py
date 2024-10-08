import torch
import torch.nn as nn
import torch.nn.functional as F

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

        self.conv1_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                      kernel_size = self.kernel_size, dilation=1,padding =0)
        
        input_size = int((self.enc_in-1)*self.input_window_size*self.num_filters)
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




    def forward(self, inputs,_x,y,_y):
        concatenated = self.process_feature(inputs)
        # flatten_input = inputs.flatten()
        x = F.relu(self.dense1(concatenated))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))
        # print(f"x to output {x.shape}")
        
        # Output layer
        output = self.output_layer(x)
        output = torch.unsqueeze(output,1)
        
        return output



class Model_arch(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.verbose = configs.verbose
        self.input_window_size = configs.seq_len
        self.kernel_size = configs.kernel_size
        self.num_filters = configs.num_kernels
        self.pred_len = configs.label_len

        self.conv1_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                      kernel_size = self.kernel_size, dilation=1,padding =0)
        
        self.conv2_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                      kernel_size = self.kernel_size, dilation=2,padding = 0)
        
        self.conv3_layers = nn.Conv1d(in_channels = 1, out_channels = self.num_filters, 
                                      kernel_size = self.kernel_size, dilation=4,padding = 0)

        self.dense1 = nn.Linear(2442 , 4096 * 4)
        self.dense2 = nn.Linear(4096 * 4, 1024 * 4)
        self.dense3 = nn.Linear(1024 * 4, 1024 * 4)
        self.dense4 = nn.Linear(1024 * 4, 64)
        self.output_layer = nn.Linear(64, self.pred_len)
        self.output_layerf = nn.Linear(16, 1)

    def process_feature(self, input_feature):
        conv1 = F.relu(self.conv1_layers(input_feature))
        conv2 = F.relu(self.conv2_layers(input_feature))
        conv3 = F.relu(self.conv3_layers(input_feature))
        # print(f"shapes {conv1.shape} {conv2.shape} {conv3.shape}")
        concatenated = torch.cat((conv1, conv2, conv3), dim=2)

        return concatenated

    def forward(self, inputs,_x,y,_y):
        # print(f"shape of input {inputs.shape}")
        # Split the input into separate features
        features = [inputs[:, i:i+1, :] for i in range(6)]
        # features = [inputs[:, :, i:i+1] for i in range(138)]

        # Process each feature separately
        processed_features = [self.process_feature(feature) for feature in features]

        # Concatenate the processed features along the feature dimension
        concatenated_features = torch.cat(processed_features, dim=2)
        # print(f"before flatten {concatenated_features.shape}")
        # Flatten the concatenated features
        # flattened_features = concatenated_features.view(concatenated_features.size(0), -1)
        # print(f"flattened {flattened_features.shape}")
        # Dense layers (MLP)
        x = F.relu(self.dense1(concatenated_features))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))

        # Output layer
        output = self.output_layer(x)
        output = output.permute(0,2,1)
        # print(f"output {output.shape}")
        output = self.output_layerf(output)
        # print(f"outputf {output.shape}")
        
        return output
