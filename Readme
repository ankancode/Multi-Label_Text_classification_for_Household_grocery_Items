
## .py Files:

preprocessing.py -
•	It converts the labels in training data in one hot encoding 
•	Splits the train_data into train.csv and validation.csv and saves the csvs in the Data folder

dataloader.py –
•	I have used torchtext to tokenize the titles, creating the vocabulary and creating the trainloader, validationloader and testloader

model.py –
•	It contain the Bi-lstm network architecture 

train_and_save.py –
•	This script is used for training the network and then saving the checkpoints of the network

test.py – 
•	This script is used to test the network on the unseen data.
•	In the Results folder -> test_results.csv is generated using test.py ( Data used is test.csv which is saved inside Data folder )

## Model Used: 
I attempted to solve the multi-label classification problem using 2 layer Bi-LSTM . ( For the full network configuration see model.py)
For the Loss Function I have used BCEWithLogitsLoss
Optimizer used : Adam

Final Loss after 12 Epoches of training:
Training Loss: 0.0049
Validation Loss: 0.0028

Model_Checkpoints folder contains the weight files.
Final result csv is saved in the Results folder -> test_results.csv. It contains the titles and probability scores for each of the labels.

## Packages Used:
PyTorch, Sklearn, Torchtext, Pandas, Numpy
