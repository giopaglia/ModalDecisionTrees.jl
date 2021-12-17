using PyCall

# py_script = read("$(py_script_path)/pipeline.py", String)

# py"""
# $(py_script)
# """
py"""

#!/usr/bin/python3

# py_script_path = "neuro-symbolic/pipeline"
sub_dir = $(py_script_path)
import ast
import os
from wasabi import msg
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
from copy import copy
from torch import nn
import numpy as np

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

				# if self.training:
				#     cls = self.softmax(self.linear(encoder_out))[:,-1,:]
				#     decoder_out = self.decoder(encoder_out, attention_mask)
				#     return decoder_out, cls
				
				return encoder_out 

import torch
from wasabi import msg
#torch.use_deterministic_algorithms(True)


def get_dataset(dataset):
		if dataset == "fingermovements":
				return "fingermovements", 2, 28
		raise Exception("You must specify a dataset (only \'fingermovements\' is available)")


model_in_use = None    # model initialization

def load_model(dataset, seed):
		global model_in_use
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		dataset, n_classes, n_attributes = get_dataset(dataset)

		path = f"{sub_dir}/assets/checkpoints/model_{dataset}_{1}_seed_{seed}.pt" 
		if not os.path.isfile(path):
				msg.warn(f"No model. Wrong seed or dataset!")
		model_in_use = BasicAutoencoder(n_attributes, n_classes) 
		model_in_use.load_state_dict(torch.load(f"{sub_dir}/assets/checkpoints/model_{dataset}_{1}_seed_{seed}.pt", map_location=device))
		model_in_use = model_in_use.to(device)
		model_in_use.eval()
		model_in_use.training = False

@torch.inference_mode()
def validation(input_serie):
		global model_in_use
		# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		
		# with torch.no_grad():
		input_ids = torch.tensor(input_serie,requires_grad = False).unsqueeze(0)#.to(device)
		mask = torch.ones(len(input_serie), requires_grad = False).unsqueeze(0)#.to(device)
		result = model_in_use(input_ids, mask)
		result = result[0,-1,:].squeeze()
		return result.detach().cpu().numpy()#.tolist()

# def get_model(dataset, seed):
# 		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 		dataset, n_classes, n_attributes = get_dataset(dataset)

# 		path = f"{sub_dir}/assets/checkpoints/model_{dataset}_{1}_seed_{seed}.pt" 
# 		if not os.path.isfile(path):
# 				msg.warn(f"No model. Wrong seed or dataset!")
# 		model_in_use = BasicAutoencoder(n_attributes, n_classes) 
# 		model_in_use.load_state_dict(torch.load(f"{sub_dir}/assets/checkpoints/model_{dataset}_{1}_seed_{seed}.pt", map_location=device))
# 		model_in_use = model_in_use.to(device)
# 		model_in_use.eval()
# 		model_in_use.training = False
# 		return model_in_use

# @torch.inference_mode()
# def validation(model_in_use, input_serie):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		
#     with torch.no_grad():
#         input_ids = torch.tensor(input_serie).unsqueeze(0).to(device)
#         mask = torch.ones(len(input_serie)).unsqueeze(0).to(device)
#         result = model_in_use(input_ids, mask)
#         result = result[0,-1,:].squeeze()
#         return result.detach().cpu().numpy().tolist()


# @torch.inference_mode()
# def validation(model_in_use, input_serie):
# 		# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 		input_ids = torch.tensor(input_serie, requires_grad = False).unsqueeze(0) #.to(device)
# 		mask = torch.ones(len(input_serie)).unsqueeze(0) #.to(device)
# 		result = model_in_use(input_ids, mask)
# 		result = result[0,-1,:].squeeze()

# 		return result.detach().cpu().numpy()#.tolist()
"""

# py"load_model"("fingermovements", 1)
# py"validation(np.zeros((2,28), dtype=np.float32))"

# model = py"get_model"("fingermovements", 1)
# py"validation"(model, randn(Float32, (2,28)))
