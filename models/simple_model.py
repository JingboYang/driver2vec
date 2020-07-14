import torch
from torch import nn


class LinearModel(nn.Module):
    def __init__(self, **kwargs):
        super(LinearModel, self).__init__()

        input_length = kwargs['input_length']
        input_channels = kwargs['input_channels']
        output_size = kwargs['output_size']

        intermediate_size = 256
        self.layers1 = nn.Sequential(nn.Linear(input_length * input_channels,
                                               intermediate_size),
                                     nn.ReLU(),
                                     nn.Linear(intermediate_size,
                                               output_size * 2),
                                     nn.Sigmoid(),
                                     )
        
        self.layers2 = nn.Sequential(nn.Linear(output_size * 2,
                                               output_size),
                                     )

    def forward(self, in_val, _1, _2, *args, **kwargs):
        in_val = in_val.contiguous().view(in_val.size()[0], -1)
        intermediate = self.layers1(in_val)
        out_val = self.layers2(intermediate)
        return out_val, {'orig': intermediate, 'pos': None, 'neg:': None}
