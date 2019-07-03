import pandas as pd
import numpy as np
import torch
import pickle
from torchtext.data import Field, TabularDataset, Iterator, BucketIterator


class BatchWrapper:
	def __init__(self, dl, x_var, y_vars):
		self.dl, self.x_var, self.y_vars = dl, x_var, y_vars # we pass in the list of attributes for x and y
	
	def __iter__(self):
		for batch in self.dl:
			x = getattr(batch, self.x_var) # we assume only one input in this wrapper
			
			if self.y_vars is not None: # we will concatenate y into a single tensor
				y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
			else:
				y = torch.zeros((1))

			yield (x, y)
	
	def __len__(self):
		return len(self.dl)


def train_val_loader(path):
	tokenize = lambda x: x.split()
	TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
	LABEL = Field(sequential=False, use_vocab=False)
	with open("Data/all_columns.txt", "rb") as fp:
		all_columns = pickle.load(fp)
	
	tv_datafields = []
	for i in all_columns:
		if i == "titles":
			tv_datafields.append((i, TEXT))
		else:
			tv_datafields.append((i, LABEL))

	trn, vld = TabularDataset.splits(
			path=path, 
			train='train.csv', validation="validation.csv",
			format='csv',
			skip_header=True, 
			fields=tv_datafields)

	TEXT.build_vocab(trn)

	with open("Data/text.pickle", "wb") as fp:
		pickle.dump(TEXT.vocab, fp)

	train_iter, val_iter = BucketIterator.splits(
		(trn, vld),batch_sizes=(32, 32),
		sort_key=lambda x: len(x.titles),
		sort_within_batch=False,
		repeat=False)

	with open("Data/labels.txt", "rb") as fp:
		labels = pickle.load(fp)

	train_dl = BatchWrapper(train_iter, "titles", labels)
	valid_dl = BatchWrapper(val_iter, "titles", labels)

	return train_dl, valid_dl, trn, vld


def test_loader(path):
	with open("Data/text.pickle", "rb") as fp:
		vocab = pickle.load(fp)
	tokenize = lambda x: x.split()
	TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
	TEXT.vocab = vocab
	tst_datafields = [("titles", TEXT)]
	tst = TabularDataset(
		path='Data/test.csv',
		format='csv',
		skip_header=True,
		fields=tst_datafields)

	test_iter = Iterator(tst, batch_size=32, sort=False,
		sort_within_batch=False, repeat=False)

	test_dl = BatchWrapper(test_iter, "titles", None)

	return test_dl, tst




