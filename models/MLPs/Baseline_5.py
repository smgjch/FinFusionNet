import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Model(nn.Module):
    def __init__(self, input_size = 138, hidden_sizes = [], output_size = 1):
        super(Model, self).__init__()
        hidden_sizes = [138,32,16,8,4,2]
        self.verbose = 0
        self.hidden_layers = nn.ModuleList()
        first = nn.Linear(138, hidden_sizes[0])
        init.kaiming_normal_(first.weight, nonlinearity='relu')
        init.constant_(first.bias, 0)


        self.hidden_layers.append(first)
        
        for i in range(1, len(hidden_sizes)):
            layer_to_append = nn.Linear(hidden_sizes[i-1], hidden_sizes[i])
            init.kaiming_normal_(layer_to_append.weight, nonlinearity='relu')
            init.constant_(layer_to_append.bias, 0)

            self.hidden_layers.append(layer_to_append)

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        init.kaiming_normal_(self.output_layer.weight, nonlinearity='relu')
        init.constant_(self.output_layer.bias, 0)

        self.relu = nn.ReLU()

    def forward(self, x,y,_x,_y):
        # print(x.shape)
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        out = self.output_layer(x)
        return out