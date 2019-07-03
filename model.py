import torch.nn as nn
from torchtext.data import Field, TabularDataset
import pickle

class Rnn_Lstm(nn.Module):
    def __init__(self, in_features, hidden_size, layer_num, output, phase='Train'):
        super(Rnn_Lstm, self).__init__()

        with open("Data/text.pickle", "rb") as fp:
            vocab = pickle.load(fp)

        self.phase = phase

        self.embedding = nn.Embedding(len(vocab), in_features)

        self.lstm = nn.LSTM(input_size=in_features,
                            hidden_size=hidden_size,
                            num_layers=layer_num,
                            batch_first=True,
                            dropout=0.5,
                            bidirectional=True)

        self.fc1 = nn.Sequential(nn.Linear(hidden_size*2, hidden_size, nn.ReLU()))
        self.fc2 = nn.Sequential(nn.Linear(hidden_size, output))
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        
                    
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.01)
                    elif 'weight' in name:
                        nn.init.xavier_normal_(param)
            elif isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, seq):
        embed_seq = self.embedding(seq)
        hidden, _ = self.lstm(embed_seq)
        feature = hidden[-1, :, :]
        feature = self.dropout(self.fc1(feature))
        
        x = self.fc2(feature)
        
        if self.phase == 'Train':
            return x
        else:
            return self.sigmoid(x)