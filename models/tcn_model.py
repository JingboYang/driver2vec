import torch.nn.functional as F
from torch import nn
from models.tcn import TemporalConvNet
import torch

class TCN(nn.Module):
    def __init__(self, **kwargs):
        super(TCN, self).__init__()
        self.input_size = kwargs['input_channels']
        self.wavelet = kwargs['wavelet']
        self.input_length = kwargs['input_length']
        output_size = kwargs['output_size']
        kernel_size = kwargs['kernel_size']
        dropout = kwargs['dropout']
        num_channels = kwargs['channel_lst']
        wavelet_output_size = kwargs['wavelet_output_size']
        num_channels  = [int(x) for x in num_channels.split(',')]
        linear_size = num_channels[-1]

        if self.wavelet:
            self.input_size = self.input_size//2
            wvlt_size = self.input_length * self.input_size // 2
            self.linear_wavelet = nn.Linear(wvlt_size, wavelet_output_size)
            linear_size += 2 * wavelet_output_size

        self.tcn = TemporalConvNet(
            self.input_size,
            num_channels,
            kernel_size=kernel_size,
            dropout=dropout)
        
        self.input_bn = nn.BatchNorm1d(linear_size)
        self.linear = nn.Linear(linear_size, output_size)
        

    def forward(self, inputs, positive, negative, need_triplet_emb=True):
        """Inputs have to have dimension (N, C_in, L_in)"""
        if self.wavelet:
            splits = torch.split(inputs, self.input_size, dim=2)
            inputs = splits[0]
            wvlt_inputs = splits[1]
            wvlt_inputs_1 = torch.split(wvlt_inputs,
                                       self.input_length // 2,
                                       dim=1)[0]
            wvlt_inputs_2 = torch.split(wvlt_inputs,
                                       self.input_length // 2,
                                       dim=1)[1]
            bsize = inputs.size()[0]
            wvlt_out1 = self.linear_wavelet(
                wvlt_inputs_1.reshape(bsize, -1, 1).squeeze())
            wvlt_out2 = self.linear_wavelet(
                wvlt_inputs_2.reshape(bsize, -1, 1).squeeze())

        inputs = inputs.permute(0, 2, 1)
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        last = y1[:, :, -1]
        
        if self.wavelet:
            last = torch.cat([last, wvlt_out1, wvlt_out2], dim=1)

        normalized = self.input_bn(last)
        o = self.linear(normalized)
        # return o, {'orig': last, 'pos': None, 'neg': None}
        return o, {'orig': normalized, 'pos': None, 'neg': None}

