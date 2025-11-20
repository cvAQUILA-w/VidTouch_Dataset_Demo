import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPFusion(nn.Module):
    def __init__(self, in_dim1, in_dim2, hidden_dim=512, out_dim=256, dropout=0.3):
        super(MLPFusion, self).__init__()
        self.fc1 = nn.Linear(in_dim1 + in_dim2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, feat1, feat2):
        x = torch.cat([feat1, feat2], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x