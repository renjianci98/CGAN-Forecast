import torch.nn as nn

from .models import register


@register('FC_Block')
class FCBlock(nn.Module):
    def __init__(self, input_dim, K, M):
        super(FCBlock, self).__init__()
        layers = []
        for i in range(K-1):
            layers.append(nn.Linear(M, M))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(
            nn.Linear(input_dim, M),
            nn.ReLU(inplace=True),
            *layers
        )

    def forward(self, X):
        X = self.block(X)
        return X
