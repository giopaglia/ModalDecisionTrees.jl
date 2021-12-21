from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
from copy import copy
from torch import nn

class AutoLayer(nn.Module):
    def __init__(self):
        super(AutoLayer, self).__init__()
        pass

    def forward(self, X, attention_mask):
        emb = X
        for layer in self.layers:
            emb = self._get_lstm_out(layer, emb, attention_mask)
        return emb

    def _get_lstm_out(self, layer, emb_input, attention_mask):
        mask = attention_mask.detach().cpu().numpy()
        mask = [elem.index(0) if 0 in elem else len(elem) for elem in mask.tolist()]
        packed_x = pack_padded_sequence(emb_input, mask, batch_first=True, enforce_sorted=False)
        lstm_out,_ = layer(packed_x)
        padded_x,_ = pad_packed_sequence(lstm_out, batch_first=True, total_length=emb_input.shape[1])
        return padded_x

class Encoder(AutoLayer):
    def __init__(self, n_attributes: int, code_size: int):
        super(Encoder, self).__init__()
        
        self.layers = torch.nn.Sequential(
            torch.nn.LSTM(n_attributes, 14, batch_first=True),
            torch.nn.LSTM(14, 7, batch_first=True),
            torch.nn.LSTM(7, code_size, batch_first=True)
            )


class Decoder(AutoLayer):
    def __init__(self, n_attributes: int, code_size: int):
        super(Decoder, self).__init__()
        
        self.layers = nn.Sequential(
            torch.nn.LSTM(code_size, 7, batch_first=True),
            torch.nn.LSTM(7, 14, batch_first=True),
            torch.nn.LSTM(14, n_attributes, batch_first=True),
            )


# basic model
class BasicAutoencoder(torch.nn.Module):
    def __init__(self, n_attributes, n_classes, code_size: int=5):
        super(BasicAutoencoder, self).__init__()
        self.encoder = Encoder(n_attributes, code_size)
        self.decoder = Decoder(n_attributes, code_size)

        self.linear = torch.nn.Linear(code_size, n_classes)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)


    def forward(self, X, attention_mask, labels=None):
        encoder_out = self.encoder(X, attention_mask) 
        cls = self.softmax(self.linear(encoder_out))[:,-1,:]

        if self.training:
            decoder_out = self.decoder(encoder_out, attention_mask)
            return decoder_out, cls
        
        return encoder_out 
