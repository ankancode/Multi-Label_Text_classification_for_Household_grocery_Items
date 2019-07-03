import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import model
import dataloader
import tqdm
import os
import shutil
import time
import argparse


def main(model, path):
	print(path)
	t1 = time.time()

	checkpoint_folder = "Model_Checkpoints"
	project_path = os.getcwd()
	save_path = os.path.join(project_path, checkpoint_folder)

	if not os.path.exists(checkpoint_folder):
		os.makedirs(checkpoint_folder)
	else:
		shutil.rmtree(save_path)
		os.makedirs(checkpoint_folder)

	in_features = 300
	hidden_size = 256
	layer_num = 2

	print("\n")
	print(" Loading Data ... ")
	print("="*30)
	print("\n")

	train_dl, valid_dl, trn, vld = dataloader.train_val_loader(path)

	print(" Got train_dataloader and validation_dataloader ")
	print("="*30)
	print("\n")

	print(" Loading LSTM Model ...")
	print("="*30)
	print("\n")
	model = model.Rnn_Lstm(in_features, hidden_size, layer_num, 391)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model.to(device)

	optimizer = optim.Adam(model.parameters(), lr=1e-2)
	criterion = nn.BCEWithLogitsLoss()

	epochs = 10

	print(" Training started ... ")
	print("="*30)
	print("\n")


	for epoch in range(1, epochs + 1):
		checkpoint_name = "checkpoint_"+ str(epoch) +".pth"
		checkpoint_save_path = os.path.join(save_path, checkpoint_name)
		running_loss = 0.0

		model.train() # turn on training mode
		for x, y in tqdm.tqdm(train_dl):
			x, y = x.to(device), y.to(device)
			optimizer.zero_grad()

			preds = model(x)
			loss = criterion(preds, y)
			loss.backward()
			optimizer.step()
			
			running_loss += loss.item() * x.size(0)
			
		epoch_loss = running_loss / len(trn)
		
		# calculate the validation loss for this epoch
		val_loss = 0.0
		model.eval() # turn on evaluation mode
		for x, y in valid_dl:
			x, y = x.to(device), y.to(device)
			preds = model(x)
			loss = criterion(preds, y)
			val_loss += loss.item() * x.size(0)

		val_loss /= len(vld)
		print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f} \n'.format(epoch, epoch_loss, val_loss))
		print("Checkpoint saved after {} epoch\n".format(epoch))
		torch.save(model.state_dict(), checkpoint_save_path)

	print("Training completed -> Finished -- {} \n".format(time.time()-t1))
	print("="*30)
	print("\n")

if __name__=='__main__':

	parser = argparse.ArgumentParser(description=" Training and Saving Model")
	parser.add_argument("-f","--folder", help="Folder Containing train.csv and validation.csv", default="Data")
	args = vars(parser.parse_args())

	main(model, args['folder'])


