import torch
import torch.nn as nn
from TCN.tcn import TemporalConvNet

class TCNRegressor(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout=0.2):
        super(TCNRegressor, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.output_size = output_size

    def forward(self, inputs):
        y1 = self.tcn(inputs)
        o = self.linear(y1[:, :, -1])
        return o

class MultiStepTCNRegressor(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, num_steps=1, dropout=0.2):
        super(MultiStepTCNRegressor, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.num_steps = num_steps
        self.output_size = output_size
        
        if num_steps == 1:
            self.linear = nn.Linear(num_channels[-1], output_size)
        else:
            self.linears = nn.ModuleList([
                nn.Linear(num_channels[-1], output_size) for _ in range(num_steps)
            ])

    def forward(self, inputs):
        y1 = self.tcn(inputs)
        
        if self.num_steps == 1:
            o = self.linear(y1[:, :, -1])
            return o
        else:
            outputs = []
            for linear in self.linears:
                outputs.append(linear(y1[:, :, -1]))
            return torch.stack(outputs, dim=1)