import torch
import torch.nn as nn
import model
import dataloader
import tqdm
import os
import shutil
import time
import pandas as pd
import numpy as np
import pickle
import argparse

def main(model, path):
	t1 = time.time()

	in_features = 300
	hidden_size = 256
	layer_num = 2

	print("\n")
	print(" Loading test Data ... ")
	print("="*30)
	print("\n")

	test_dl, tst = dataloader.test_loader("Data/test.csv")

	print(" Got test_dataloader ... ")
	print("="*30)
	print("\n")

	print(" Loading LSTM Model ...")
	print("="*30)
	print("\n")
	model = model.Rnn_Lstm(in_features, hidden_size, layer_num, 391, phase='Test')

	print(" Loading Weights on the Model ...")
	print("="*30)
	print("\n")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model.to(device)

	state_dict = torch.load('Model_Checkpoints/checkpoint_5.pth')
	model.load_state_dict(state_dict)

	model.eval()

	print(" Predicting on the test data ...")
	print("="*30)
	print("\n")

	predictions = []
	
	for x, _ in tqdm.tqdm(test_dl):
		x = x.to(device)
		preds = model(x)
		predictions.extend(preds.cpu().detach().numpy())

	with open("Data/labels.txt", "rb") as fp:
		labels = pickle.load(fp)


	test_df = pd.read_csv("Data/test.csv")
	result_df = pd.DataFrame(data=predictions, columns=labels)
	test_results = pd.concat([test_df, result_df], axis=1)

	print("\n Saving Results to test_results.csv .")
	print("="*30)
	print("\n")

	result_folder = "Results"
	project_path = os.getcwd()
	save_path = os.path.join(project_path, result_folder)

	if not os.path.exists(result_folder):
		os.makedirs(result_folder)
	else:
		shutil.rmtree(save_path)
		os.makedirs(result_folder)

	test_results.to_csv(save_path+'/test_results.csv', index=False)

	print("Completed \n")


if __name__=='__main__':

	parser = argparse.ArgumentParser(description=" Testing Model")
	parser.add_argument("-t","--testpath", help=" Path to test.csv ", default="Data/test.csv")
	args = vars(parser.parse_args())

	main(model, args['testpath'])

	    
