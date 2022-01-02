#!/usr/bin/python3
import ast
import os
import pandas as pd
import numpy as np
from model import Seq2Seq
import torch
#torch.use_deterministic_algorithms(True)


model = None    # model initialization

def load_model(dataset, seed, attribute, code, limit):
		""" Carica modello
		
		Args:
				dataset: nome del dataset (tutto minuscolo)
				seed: numero del seed
				attribute: il numero dell'attributo di cui si vuole il modello (si parte da 0)
				limit: 1 se si vogliono considerare tutti i sottointervalli

		Returns:
				None
		"""
		
		_dataset = dataset.lower()
		
		if limit != 1 and limit != 100:
				raise Exception("Limit must be 1 or 100.")
		global model
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		path = f"assets/checkpoints/{_dataset}_code_{code}/model_code_{code}_{_dataset}_limit_{limit}_seed_{seed}_attr_{attribute}.pt"

		if not os.path.isfile(path):
				raise Exception(f"No model found {path}!")
		model = Seq2Seq(device, code, input_dim = 1, hid_dim = code, dropout=0.25, emb_dim=64) 
		model.load_state_dict(torch.load(path, map_location=device))
		model = model.to(device)



@torch.inference_mode()
def validation(input_serie):
		""" Ritorna la rappresentazione di dimensione 4 per ogni istante temporale

		Args:
				input_serie: str, serie temporale di dimensione 1xn
		
		"""
		global model

		model.eval()
		# input_serie = ast.literal_eval(input_serie)
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		
		with torch.no_grad():
				input_ids = torch.tensor(input_serie).to(device)
				mask = torch.ones(1,input_ids.shape[1])
				result = model(input_ids, None, mask)
				# print(result.numpy().tolist())
				return result.numpy()

def produce_npz_flattened(dataset, seeds, code_size):
	X = np.load(f'../neuro-symbolic-MTSC-data/stump_with_memoization,TestOp_80,{dataset},(false,false,"interval"),missing-X.npy').astype(np.float32)
	n_points, n_attributes, n_instances = X.shape
	print("n_points: ",     n_points)
	print("n_attributes: ", n_attributes)
	print("n_instances: ",  n_instances)

	for seed in seeds:
		flattened_filename = f"flattened_{dataset}-{seed}-{code_size}-X.npy"
		if os.path.isfile(flattened_filename):
			print("Skipping: ", flattened_filename)
		else:
			flattened_array = np.ndarray((code_size, n_attributes, n_instances))
			for attribute in range(n_attributes):
				print(f"Attribute {attribute}/{n_attributes}")
				load_model(dataset, seed, attribute, code_size, 100)
				for instance in range(n_instances):
					# print(X[:,attribute,instance].shape)
					# print(np.expand_dims(X[:,attribute,instance], axis=0).transpose().shape)
					# res = validation(np.expand_dims(X[:,attribute,instance], axis=0).transpose())
					res = validation([X[:,attribute,instance].tolist()])
					# print(res)
					flattened_array[:,attribute,instance] = res
			# print(flattened_array)
			# print(flattened_array.shape)
			print(f"Saving {flattened_filename}...")
			np.save(flattened_filename, flattened_array)

def produce_npz_fmd(dataset, seeds, code_size):
	X = np.load(f'../neuro-symbolic-MTSC-data/stump_with_memoization,TestOp_80,{dataset},(false,false,"interval"),missing-X.npy').astype(np.float32)
	n_points, n_attributes, n_instances = X.shape
	print("n_points: ",     n_points)
	print("n_attributes: ", n_attributes)
	print("n_instances: ",  n_instances)

	for seed in seeds:
		fmd_filename = f"fmd_{dataset}-{seed}-{code_size}-X.npy"
		if os.path.isfile(fmd_filename):
			print("Skipping: ", fmd_filename)
		else:
			fmd_array = np.ndarray((code_size,n_points,n_points+1,n_attributes, n_instances))
			for attribute in range(n_attributes):
				print(f"Attribute {attribute}/{n_attributes}")
				load_model(dataset, seed, attribute, code_size, 1)
				for x in range(n_points):
					for y in range(x+1,n_points+1):
						for instance in range(n_instances):
							# print(str(x)+":"+str(y))
							# print(X[x:y,attribute,instance].shape)
							# print(np.expand_dims(X[x:y,attribute,instance], axis=0).shape)
							# print(np.expand_dims(X[x:y,attribute,instance], axis=0).transpose().shape)
							# res = validation(np.expand_dims(X[x:y,attribute,instance], axis=0).transpose())
							# print(res)
							# res = validation(np.expand_dims(X[x:y,attribute,instance], axis=0).transpose())
							res = validation([X[x:y,attribute,instance].tolist()])
							# print(res)
							fmd_array[:,x,y,attribute, instance] = res
			print(f"Saving {fmd_filename}...")
			np.save(fmd_filename, fmd_array)
			print()

################################################################################
################################################################################
################################################################################

# load_model("Libras", 1, 0, 1)
print("Testing a RacketSports model...")
load_model("RacketSports", 1, 0, 1, 1)
# load_model("RacketSports", 1, 0, 2, 1)
print("Result:", validation([[1,2,3,1,2,3,4.]]))
print("Result:", validation([[1],[2],[3],[1],[2],[3],[4.]]))
print()
# load_model("RacketSports", 1, 0, 4, 1)
print("Testing a Libras model...")
load_model("Libras", 1, 0, 1, 1)
print("Result:", validation([[1.]]))
print()


################################################################################
################################################################################
################################################################################

for code_size in [1]: # ,2,4]:
	for dataset in ["RacketSports", "Libras", "NATOPS"]:
		seeds = [1,2,3,4,5]
		produce_npz_flattened(dataset, seeds, code_size)

for code_size in [1]: # ,2,4]:
	for dataset in ["RacketSports", "Libras", "NATOPS"]:
		seeds = [1,2,3,4,5]
		produce_npz_fmd(dataset, seeds, code_size)
