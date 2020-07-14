import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):
    """
    define RNN layer
    """

#    def __init__(self, vocab_size, embed_size, num_output, rnn_model='LSTM', use_last=True, embedding_tensor=None,
# padding_index=0, hidden_size=64, num_layers=1, batch_first=True):
    def __init__(self, **kwargs):
        """

        Args:
            vocab_size: vocab size
            embed_size: embedding size
            num_output: number of output (classes)
            rnn_model:  LSTM or GRU
            use_last:  bool
            embedding_tensor:
            padding_index:
            hidden_size: hidden size of rnn module
            num_layers:  number of layers in rnn module
            batch_first: batch first option
        """

        super(RNN, self).__init__()

        self.input_size = kwargs['input_channels']
        num_output = kwargs['output_size']
        rnn_model = kwargs['rnn_model']
        use_last = kwargs['rnn_use_last']
        hidden_size = kwargs['rnn_hidden_size']
        num_layers = kwargs['rnn_num_layers']
        dropout = kwargs['dropout']
        self.input_length = kwargs['input_length']
        self.wavelet = kwargs['wavelet']
        wavelet_output_size = kwargs['wavelet_output_size']


        self.use_last = use_last
        linear_size = hidden_size
        self.drop_en = nn.Dropout(p=dropout)
        if self.wavelet:
            self.input_size = self.input_size//2
            wvlt_size = self.input_length * self.input_size // 2
            self.linear_wavelet = nn.Linear(wvlt_size, wavelet_output_size)
            linear_size += 2 * wavelet_output_size

        # rnn module, not bidirectional since driving data is time series
        if rnn_model == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=self.input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
                bidirectional=False)
        elif rnn_model == 'GRU':
            self.rnn = nn.GRU(
                input_size=self.input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
                bidirectional=False)
        else:
            raise LookupError(' only support LSTM and GRU')

        self.bn2 = nn.BatchNorm1d(linear_size)
        self.fc = nn.Linear(linear_size, num_output)

    def forward(self, x, _1, _2, need_triplet_emb=True):
        '''
        Args:
            x: (batch, time_step, input_size)

        Returns:
            num_output size
        '''
#        packed_input = pack_padded_sequence(x_embed, seq_lengths.cpu().numpy(),batch_first=True)

        # r_out shape (batch, time_step, output_size)
        # None is for initial hidden state
#        packed_output, ht = self.rnn(packed_input, None)
#        out_rnn, _ = pad_packed_sequence(packed_output, batch_first=True)
        if self.wavelet:
            splits = torch.split(x, self.input_size, dim=2)
            x = splits[0]
            wvlt_inputs = splits[1]
            wvlt_inputs_1 = torch.split(wvlt_inputs,
                                       self.input_length // 2,
                                       dim=1)[0]
            wvlt_inputs_2 = torch.split(wvlt_inputs,
                                       self.input_length // 2,
                                       dim=1)[1]
            bsize = x.size()[0]
            wvlt_out1 = self.linear_wavelet(
                wvlt_inputs_1.reshape(bsize, -1, 1).squeeze())
            wvlt_out2 = self.linear_wavelet(
                wvlt_inputs_2.reshape(bsize, -1, 1).squeeze())
        x_embed = self.drop_en(x)

        out_rnn, _ = self.rnn(x_embed, None)
        row_indices = torch.arange(0, x.size(0)).long()
        col_indices = self.input_size - 1
        if next(self.parameters()).is_cuda:
            row_indices = row_indices.cuda()
            col_indices = col_indices

        if self.use_last:
            last_tensor = out_rnn[row_indices, col_indices, :]
        else:
            # use mean
            last_tensor = out_rnn[row_indices, :, :]
            last_tensor = torch.mean(last_tensor, dim=1)

        if self.wavelet:
            fc_input = torch.cat([last_tensor, wvlt_out1, wvlt_out2], dim=1)
        fc_input = self.bn2(fc_input)

        out = self.fc(fc_input)
        return out, {'orig': fc_input, 'pos': None, 'neg:': None}
