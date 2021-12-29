#!/usr/bin/python3
import ast
import os
from wasabi import msg
import pandas as pd
import numpy as np
from model import TST
import torch
#torch.use_deterministic_algorithms(True)

model = None    # model initialization

def load_model(dataset, seed, attribute, limit, code_size):
    """ Carica modello

    Args:
        dataset: nome del dataset (tutto minuscolo)
        seed: numero del seed
        attribute: il numero dell'attributo di cui si vuole il modello (si parte da 0)
        limit: 1 se si vogliono considerare tutti i sottointervalli

    Returns:
        None
    """
    if limit != 1 and limit != 100:
        raise Exception("Limit must be 1 or 100.")
    global model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(dataset)

    path = f"assets/checkpoints/{dataset}-{code_size}/model_auto_transformers_{dataset}_{limit}_seed_{seed}_attr_{attribute}.pt" 
    if not os.path.isfile(path):
        raise Exception(f"No model found {path}. Wrong seed or dataset: !")
    model = TST(code_size) 
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)

def get_dataset(dataset):
    """ Returns datasets' properties.
    
    Returns:
        - correct dataset name
    """
    dataset = dataset.lower()
    if dataset == "fingermovements":
        return "FingerMovements"
    if dataset == "lsst":
        return "LSST"
    if dataset == "libras":
        return "Libras"
    if dataset == "natops":
        return "NATOPS"
    if dataset == "racketsports":
        return "RacketSports"
    raise Exception("No dataset selected")

@torch.inference_mode()
def validation(input_serie):
    """ Ritorna la rappresentazione di dimensione 4 per ogni istante temporale

    Args:
        input_serie: str, serie temporale di dimensione nx1
    
    """
    global model

    model.eval()
    # input_serie = ast.literal_eval(input_serie)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        input_ids = torch.tensor(input_serie).unsqueeze(0).to(device)
        # print(input_ids)
        # print(input_ids.shape)
        result = model(input_ids, test=True)
        return result.detach().cpu().numpy().tolist()[0]

# load_model("Libras", 1, 0, 1)
load_model("RacketSports", 1, 0, 1, 2)
print(validation([[1],[2],[3],[4.]]))
load_model("RacketSports", 1, 0, 1, 4)
print(validation([[1.]]))

def produce_npz(dataset, seed, code_size):
	X = np.load(f'../neuro-symbolic-MTSC-data/stump_with_memoization,TestOp_80,{dataset},(false,false,"interval"),missing-X.npy').astype(np.float32)
	n_points, n_attributes, n_instances = X.shape
	print("n_points: ",     n_points)
	print("n_attributes: ", n_attributes)
	print("n_instances: ",  n_instances)

	flattened_filename = f"flattened_{dataset}-{seed}-{code_size}-X.npy"
	if os.path.isfile(flattened_filename):
		print("Skipping: ", flattened_filename)
	else:
		flattened_array = np.ndarray((code_size,n_attributes, n_instances))
		for attribute in range(n_attributes):
			print(f"Attribute {attribute}/{n_attributes}")
			load_model(dataset, seed, attribute, 100, code_size)
			for instance in range(n_instances):
				# print(X[:,attribute,instance].shape)
				# print(np.expand_dims(X[:,attribute,instance], axis=0).transpose().shape)
				flattened_array[:,attribute,instance] = validation(np.expand_dims(X[:,attribute,instance], axis=0).transpose())
		# print(flattened_array)
		# print(flattened_array.shape)
		print(f"Saving {flattened_filename}...")
		np.save(flattened_filename, flattened_array)

	fmd_filename = f"fmd_{dataset}-{seed}-{code_size}-X.npy"
	if os.path.isfile(fmd_filename):
		print("Skipping: ", fmd_filename)
	else:
		fmd_array = np.ndarray((code_size,n_points,n_points+1,n_attributes, n_instances))
		for attribute in range(n_attributes):
			print(f"Attribute {attribute}/{n_attributes}")
			load_model(dataset, seed, attribute, 1, code_size)
			for x in range(n_points):
				for y in range(x+1,n_points+1):
					for instance in range(n_instances):
						# print(str(x)+":"+str(y))
						# print(X[x:y,attribute,instance].shape)
						# print(np.expand_dims(X[x:y,attribute,instance], axis=0).shape)
						# print(np.expand_dims(X[x:y,attribute,instance], axis=0).transpose().shape)
						# res = validation(np.expand_dims(X[x:y,attribute,instance], axis=0).transpose())
						# print(res)
						fmd_array[:,x,y,attribute, instance] = validation(np.expand_dims(X[x:y,attribute,instance], axis=0).transpose())
		print(f"Saving {fmd_filename}...")
		np.save(fmd_filename, fmd_array)

for code_size in [4]: # , 2]:
	for dataset in ["RacketSports", "Libras"]: # , "FingerMovements", "LSST", "NATOPS"]:
		for seed in [1,2,3,4,5]:
			produce_npz(dataset, seed, code_size)
			# break
