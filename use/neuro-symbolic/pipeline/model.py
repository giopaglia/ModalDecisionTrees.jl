from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
from torch import nn
from copy import copy
import torch
from torch import nn
import random
import os
import numpy as np

MASTER_SEED = 42
torch.manual_seed(MASTER_SEED)
os.environ['PYTHONHASHSEED'] = str(MASTER_SEED)
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)
torch.cuda.manual_seed_all(MASTER_SEED)

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
            x = x.unsqueeze(-1)
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

class Decoder2(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout, code):
        super().__init__()

        self.hid_dim = hid_dim
        
        self.embedding = Time2Vec(input_dim,emb_dim)#LinearEmbedding(input_dim, emb_dim, dropout)
        self.linear = nn.Linear(code, hid_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, input_dim)

    def forward(self, src, hidden):
        embedded = self.embedding(src)
        context = hidden.squeeze(0).unsqueeze(1) #N
        context = context.repeat(1,embedded.shape[1],1) #N
        hiddens = torch.cat((context, embedded), dim=-1) #N
        # prima prendeva embedded
        outputs, _ = self.rnn(hiddens, hidden)#self.linear(hidden)) #no cell state!
        
        # outputs sono gli hidden di ogni passo
        # il context è quello prodotto dall'encoder
        # embedded è la target shiftata
        #print(context.shape)
        #print(embedded.shape)
        #print(outputs.shape)
        output = torch.cat((embedded, outputs, context), 
                           dim = -1)
        
        return self.fc_out(output)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout, code):
        super().__init__()

        self.hid_dim = hid_dim
        
        #self.linear = nn.Linear(hid_dim, code)
        self.embedding = Time2Vec(input_dim, emb_dim)#LinearEmbedding(input_dim, emb_dim, dropout)
        
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
        
    def forward(self, src, mask):
        
        #src = [src len, batch size]
        #print("e1", src.shape)
        embedded = self.embedding(src)
        
        #print("e2", embedded.shape)
        #embedded = [src len, batch size, emb dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, mask, batch_first=True,enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed_embedded) #no cell state!
        outputs,_ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True,total_length=embedded.shape[1])
        #print("e3", hidden.shape)
        #print("e1", outputs)
        #print("------")
        #print("h1", hidden)
        #print("------")
        #print("h2", hidden2)
        #print("------")
        ##outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden#self.linear(hidden)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.output_dim = output_dim
        
        self.embedding = LinearEmbedding(output_dim, emb_dim, dropout)
        
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, batch_first=True)
        
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        
    def forward(self, input, hidden, context):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #context = [n layers * n directions, batch size, hid dim]
        
        #n layers and n directions in the decoder will both always be 1, therefore:
        #hidden = [1, batch size, hid dim]
        #context = [1, batch size, hid dim]
        #hidden = hidden.squeeze().unsqueeze(1)
        context = context.squeeze().unsqueeze(1)

        #print("d1", input.shape)
        input = input.unsqueeze(-1)
        #print("d2", input.shape) 
        #input = [1, batch size]
        
        embedded = self.embedding(input)
        #print("d3", embedded.shape) 
        
        #embedded = [1, batch size, emb dim]
                
        emb_con = torch.cat((embedded, context), dim = 2)
            
        #emb_con = [1, batch size, emb dim + hid dim]
            
        output, hidden = self.rnn(emb_con, hidden)
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        #seq len, n layers and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        #print("d4", embedded.shape)
        #print("d5", hidden.shape)
        #print("d6", context.shape)

        embedded = embedded.squeeze()
        context = context.squeeze()
        output = torch.cat((embedded, hidden.squeeze(0), context), 
                           dim = 1)
        
        #output = [batch size, emb dim + hid dim * 2]
        
        prediction = self.fc_out(output)
        
        #prediction = [batch size, output dim]
        #print(prediction)
        #print("d7", prediction.shape)
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, device, code, input_dim = 1, hid_dim=2, dropout=0.25, emb_dim=16):
        super().__init__()
        self.encoder = Encoder(input_dim, emb_dim, hid_dim, dropout, code)
        self.decoder = Decoder2(input_dim, emb_dim, hid_dim, dropout, code)
        self.device = device
        
    def forward(self, src, trg, mask, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        #batch_size = trg.shape[0]
        #trg_len = trg.shape[1]
        #trg_vocab_size = 1#self.decoder.output_dim
        
        #tensor to store decoder outputs
        #outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        mask = [elem.index(0) if 0 in elem else len(elem) for elem in mask.numpy().tolist()]
        context = self.encoder(src, mask)
        if trg != None:
            out = self.decoder(trg, context)
            return out[:,:-1,:]
        else:
            return context.squeeze()
