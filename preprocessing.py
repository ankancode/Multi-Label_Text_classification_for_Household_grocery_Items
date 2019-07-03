import pandas as pd
import numpy as np
import torch
from torchtext.data import Field, TabularDataset
from sklearn.model_selection import train_test_split
import ast
import pickle
import re
import os
import shutil

df = pd.read_csv("Raw_Data/training_data.csv")

df['titles'] = df['titles'].map(lambda x: re.sub(r'\W+', ' ', x))

df['titles'] = df['titles'].map(lambda x: re.sub(r'\d+', ' ', x))

all_labels = []
for i in df.labels:
    all_labels.extend(ast.literal_eval(i))

unique_labels = list(set(all_labels))
dict_label = {unique_labels[i]:i for i in range(0, len(unique_labels))}
id_to_label = dict((v, k) for k, v in dict_label.items())


data_folder = "Data"
project_path = os.getcwd()
save_path = os.path.join(project_path, data_folder)

if not os.path.exists(data_folder):
	os.makedirs(data_folder)
else:
	shutil.rmtree(save_path)
	os.makedirs(data_folder)
	

with open("Data/labels.txt", "wb") as fp:
    pickle.dump(unique_labels, fp)

with open("Data/id_to_label.txt", "wb") as fp:
    pickle.dump(id_to_label, fp)

list_of_multi_labels = []
for i in df.labels:
    l = [0 for i in range(len(unique_labels))]
    label_elements = ast.literal_eval(i)
    for j in label_elements:
        l[dict_label[j]] = 1
    list_of_multi_labels.append(tuple(l))

df_multi_label_one_hot = pd.DataFrame.from_records(list_of_multi_labels, columns=unique_labels)

df = df.drop(labels=['labels'], axis=1, inplace=False)
df_new = pd.concat([df, df_multi_label_one_hot], axis=1)

with open("Data/all_columns.txt", "wb") as fp:
    pickle.dump(list(df_new.columns), fp)

train, validation = train_test_split(df_new, test_size=0.1)

train.to_csv('Data/train.csv', mode='w', index=False)
validation.to_csv('Data/validation.csv', mode='w', index=False)