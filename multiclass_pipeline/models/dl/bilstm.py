# multiclass_pipeline/models/dl/bilstm.py
import torch
import torch.nn as nn
from configs.config import VOCAB_SIZE

class BiLSTMModelMulti(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=128, hidden_dim=64, num_classes=5, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                            bidirectional=True, num_layers=2, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        # Use only the last layer's hidden states for cat
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(self.dropout(h))
