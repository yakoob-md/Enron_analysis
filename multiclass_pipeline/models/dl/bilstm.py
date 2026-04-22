# multiclass_pipeline/models/dl/bilstm.py
import torch
import torch.nn as nn
from configs.config import VOCAB_SIZE

class BiLSTMModelMulti(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, 128)
        self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        x = torch.cat((h[-2], h[-1]), dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
