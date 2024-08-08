import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxAbsolutePooling1D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super(MaxAbsolutePooling1D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        abs_x = torch.abs(x)
        
        pooled_abs, indices = F.max_pool1d(abs_x, self.kernel_size, self.stride, 
                                           self.padding, self.dilation, 
                                           return_indices=True)
        
        batch_size, channels, output_length = pooled_abs.shape
        batch_indices = torch.arange(batch_size).view(-1, 1, 1).expand(-1, channels, output_length)
        channel_indices = torch.arange(channels).view(1, -1, 1).expand(batch_size, -1, output_length)
        
        output = x[batch_indices, channel_indices, indices]
        
        return output