import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        inputsize = configs.enc_in * configs.seq_len
        output_size = configs.label_len
        self.args = configs
        hidden_sizes = [20,15,10,5]
        self.verbose = 0
        self.hidden_layers = nn.ModuleList()
        first = nn.Linear(inputsize, hidden_sizes[0])
        # init.kaiming_normal_(first.weight, nonlinearity='relu')
        # init.constant_(first.bias, 0)


        self.hidden_layers.append(first)
        
        for i in range(1, len(hidden_sizes)):
            layer_to_append = nn.Linear(hidden_sizes[i-1], hidden_sizes[i])
            # init.kaiming_normal_(layer_to_append.weight, nonlinearity='relu')
            # init.constant_(layer_to_append.bias, 0)

            self.hidden_layers.append(layer_to_append)

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        # init.kaiming_normal_(self.output_layer.weight, nonlinearity='relu')
        # init.constant_(self.output_layer.bias, 0)

        self.relu = nn.ReLU()

    def forward(self, x,y,_x,_y):
        # print(x.shape)

        x = x.reshape(self.args.batch_size,-1)
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        out = self.output_layer(x).reshape(self.args.batch_size,1,-1)
        return out