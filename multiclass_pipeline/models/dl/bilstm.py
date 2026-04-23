# multiclass_pipeline/models/dl/bilstm.py
# FIXED vs original:
#   - forward() now returns raw logits (no sigmoid — softmax is applied by loss)
#   - num_layers dropout properly connected
#   - Layer norm added
#   - h concatenation fixed for num_layers > 1

import torch
import torch.nn as nn

try:
    from configs.config import VOCAB_SIZE
except ImportError:
    VOCAB_SIZE = 10000


class BiLSTMModelMulti(nn.Module):
    """
    Multiclass BiLSTM.
    Output: raw logits of shape (batch, num_classes).
    Use with CrossEntropyLoss or MulticlassFocalLoss (NOT BCEWithLogitsLoss).
    """
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=128,
                 hidden_dim=128, num_classes=5, dropout=0.4):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm       = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout          # dropout between layers
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout    = nn.Dropout(dropout)
        self.fc         = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        # h shape: (num_layers * 2, batch, hidden_dim)
        # Take only the LAST layer's forward + backward states
        h = torch.cat([h[-2], h[-1]], dim=1)  # (batch, hidden*2)
        h = self.layer_norm(h)
        h = self.dropout(h)
        return self.fc(h)   # raw logits — DO NOT apply sigmoid here