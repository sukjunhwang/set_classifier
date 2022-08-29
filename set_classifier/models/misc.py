import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def js_div(v1, v2):
    p = (v1[:, None] + v2[None]) / 2
    kl_div1 = v1[:, None] * torch.log(v1[:, None] / p)
    kl_div2 = v2[None] * torch.log(v2[None] / p)

    return (kl_div1.sum(dim=2) + kl_div2.sum(dim=2)) / 2
