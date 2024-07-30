
import torch
import torch.nn as nn
import torch.nn.functional as F



class Model(nn.Module):
# class Model(nn.Module):
    def __init__(self, input_size=138, hidden_sizes=None, output_size=1):
        super(Model, self).__init__()
        input_size=138
        output_size=1
        hidden_sizes = [5,5,5,5, 5, 5, 5,5]
        # hidden_sizes = [138, 256, 128, 64, 32, 16, 8, 4, 2]

        self.verbose = 0
        
      
        self.hidden_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()  # Adding batch normalization layers
        self.residual_layers = []
        self.projection_layers = nn.ModuleList()


        for i in range(0, len(hidden_sizes)-1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.batch_norm_layers.append(nn.BatchNorm1d([hidden_sizes[i+1]]))  # Add batch norm layer after each linear layer
            # Add projection layers at the residual points
            if (i + 1) % 5 == 0:
                self.projection_layers.append(nn.Linear(hidden_sizes[i-4], hidden_sizes[i+1]))
                self.residual_layers.append(i)
        
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.relu = nn.ReLU()

    def forward(self, x,y,_x,_y):
        residual = x
        projection_idx = 0
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)

            x = x.permute(0,2,1)
            x = self.batch_norm_layers[i](x)  # Apply batch normalization
            x = x.permute(0,2,1)
            x = self.relu(x)
            
            if i in self.residual_layers:
                residual = self.projection_layers[projection_idx](residual)
                x = x + residual
                residual = x
                projection_idx += 1
        
        out = self.output_layer(x)
        return out