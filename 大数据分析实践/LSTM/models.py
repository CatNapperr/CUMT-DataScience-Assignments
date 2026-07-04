import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_dim=16, seq_len=3, num_classes=4):
        super().__init__()
        in_features = seq_len * input_dim
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x.flatten(1))


class LSTMModel(nn.Module): 

    def __init__(self, input_dim=16, hidden1=64, hidden2=32, num_classes=4):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.fc = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        return self.fc(out[:, -1, :])


class LSTMAttention(nn.Module):

    def __init__(self, input_dim=16, hidden=64, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        self.attn = nn.Linear(hidden, 1)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x, return_attention=False):
        out, _ = self.lstm(x)                    # (N, 3, hidden)
        scores = self.attn(out).squeeze(-1)      # (N, 3)
        weights = torch.softmax(scores, dim=-1)  # (N, 3)
        context = (out * weights.unsqueeze(-1)).sum(dim=1)  # (N, hidden)
        logits = self.fc(context)
        if return_attention:
            return logits, weights
        return logits
