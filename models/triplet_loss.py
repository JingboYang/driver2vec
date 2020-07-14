import torch.nn.functional as F
from torch import nn
import models


class TripletLoss(nn.Module):
    def __init__(self, **kwargs):
        super(TripletLoss, self).__init__()

        self.embedding_model = \
            models.__dict__[kwargs['model_name']](**kwargs)
        self.activation = nn.Sigmoid()
        # TODO Eventually remove the useless linear layer
        # self.linear = nn.Linear(num_channels[-1], output_size)
        # batchnorm for input, pos and neg seq
        print('Triplet initialized!')

    def forward(self, inputs, positive, negative, need_triplet_emb=True):
        """Inputs have to have dimension (N, C_in, L_in)"""
        o, cur_emb_out = self.embedding_model(inputs, None, None)  # input should have dimension (N, C, L)
        last = self.activation(cur_emb_out['orig'])

        if need_triplet_emb:
            o, pos_emb_out = self.embedding_model(positive, None, None)
            o, neg_emb_out = self.embedding_model(negative, None, None)    
            plast = self.activation(pos_emb_out['orig'])
            nlast = self.activation(neg_emb_out['orig'])
        else:
            plast = None
            nlast = None
        
        return o, {'orig': last, 'pos': plast, 'neg': nlast}