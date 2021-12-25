from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
from torch import nn
from copy import copy
import torch
from torch import nn


class Time2Vec(nn.Module):
    def __init__(self, input_dim, embed_dim, act_function=torch.sin):
        assert embed_dim % input_dim == 0
        super(Time2Vec, self).__init__()
        self.enabled = embed_dim > 0
        if self.enabled:
            # // is useless because we know that embde_dim%input_dim==0
            self.embed_dim = embed_dim // input_dim
            self.input_dim = input_dim
            self.embed_weight = nn.parameter.Parameter(
                torch.randn(self.input_dim, self.embed_dim)
            )
            self.embed_bias = nn.parameter.Parameter(torch.randn(self.embed_dim))
            self.act_function = act_function

    def forward(self, x):
        if self.enabled:
            # size of x = [bs, sample, input_dim]
            x = torch.diag_embed(x)
            x_affine = torch.matmul(x, self.embed_weight) + self.embed_bias
            # size of x_affine = [bs, sample, embed_dim]
            x_affine_0, x_affine_remain = torch.split(
                x_affine, [1, self.embed_dim - 1], dim=-1
            )
            x_affine_remain = self.act_function(x_affine_remain)
            x_output = torch.cat([x_affine_0, x_affine_remain], dim=-1)
            x_output = x_output.view(x_output.size(0), x_output.size(1), -1)
        else:
            x_output = x
        return x_output

class TEncoder(torch.nn.Module):
    def __init__(self, n_attributes, emb_size, code_size):
        super(TEncoder,self).__init__()
        # from (B,N,n_attributes) to (B,N,emb_size)
        self.embedding_layer = Time2Vec(input_dim=n_attributes, embed_dim=emb_size)
        self.encoder = torch.nn.TransformerEncoderLayer(d_model=emb_size, nhead=2, batch_first=True, dropout=0.25)
        self.fc_enc = torch.nn.Linear(emb_size, code_size) # (B,L,E)

    def forward(self, x, mask):
        # print("1", x.shape)
        x = self.embedding_layer(x)
        # print("2", x.shape)
        x = self.encoder(src=x, src_key_padding_mask=mask)
        # print("3", x.shape)
        x = self.fc_enc(x)
        # print("4", x.shape)
        x = x.squeeze(0)
        x = x.squeeze(1)
        # print("5", x.shape)
        return x

class TDecoder(torch.nn.Module,):
    def __init__(self, n_attributes, emb_size, code_size):
        super(TDecoder,self).__init__()
        # from (B,N,n_attributes) to (B,N,emb_size)
        self.fc_dec1 = torch.nn.Linear(code_size, emb_size) # (B,L,E)
        self.decoder = torch.nn.TransformerEncoderLayer(d_model=emb_size, nhead=2, batch_first=True, dropout=0.25)
        self.fc_dec2 = torch.nn.Linear(emb_size, n_attributes) # (B,L,E)
        self.fc_dec3 = torch.nn.Linear(emb_size, n_attributes) # (B,L,E)

    def forward(self, x, mask):
        x = self.fc_dec1(x)
        x = self.decoder(src=x, src_key_padding_mask=mask)
        x = self.fc_dec2(x)
        return x


class TST(torch.nn.Module):
    def __init__(self):
        """
            Encoder => (B,N,n_attributes) --> (B,N,emb_size) --> (B,N,emb_size) --> (B,N,code_size)
                                                        |
                                                        v
            Decoder => (B,N,code_size) --> (B,N,emb_size) --> (B,N,emb_size) --> (B,N,n_attributes)

        """
        super(TST, self).__init__()
        n_attributes: int = 1
        code_size: int = 2
        emb_size: int = 168
        heads: int = 2
        assert emb_size % heads == 0 and code_size % heads == 0, "Wrong number of heads"

        self.encoder = TEncoder(n_attributes, emb_size, code_size)
        self.decoder = TDecoder(n_attributes, emb_size, code_size)
    
    def forward(self, x, mask=None, test=False):
        # unsqueeze added because batch_size == 1, same as removing the squeeze in the encoder
        # print(self.encoder(x, mask).shape)
        x = self.encoder(x, mask).unsqueeze(0)
        # print("6", x.shape)
        if test:
            return x[:,0,:]
        series_len = x.shape[1]
        x = x[:,0,:].unsqueeze(1).repeat(1,series_len,1)
        x = self.decoder(x,mask)
        return x

    @property
    def name(self):
        return "auto_transformers"

