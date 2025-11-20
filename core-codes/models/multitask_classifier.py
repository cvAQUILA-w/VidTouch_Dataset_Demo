import torch
import torch.nn as nn
import math

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class MultiTaskClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_weave, num_material, num_usage, num_features, drop_path_rate=0.1):
        super(MultiTaskClassifier, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            DropPath(drop_path_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            DropPath(drop_path_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.head_weave = nn.Linear(hidden_dim, num_weave)
        self.head_material = nn.Linear(hidden_dim, num_material)
        self.head_usage = nn.Linear(hidden_dim, num_usage)
        self.head_feature = nn.Linear(hidden_dim, num_features)  # multitask（sigmoid）

    def forward(self, x):
        x = self.encoder(x)
        pred_weave = self.head_weave(x)
        pred_material = self.head_material(x)
        pred_usage = self.head_usage(x)
        pred_feature = self.head_feature(x)
        return pred_weave, pred_material, pred_usage, pred_feature