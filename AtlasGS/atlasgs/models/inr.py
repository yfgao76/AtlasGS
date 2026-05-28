import math

import torch
import torch.nn as nn


def fourier_encode(x, num_frequencies=6):
    freqs = 2.0 ** torch.arange(num_frequencies, device=x.device)
    x_exp = x[..., None] * freqs
    return torch.cat([torch.sin(math.pi * x_exp), torch.cos(math.pi * x_exp)], dim=-1)


class INR(nn.Module):
    def __init__(self, in_dim=3, hidden=128, depth=6, fourier_freqs=6):
        super().__init__()
        self.fourier_freqs = fourier_freqs
        feat_dim = in_dim + in_dim * 2 * fourier_freqs

        layers = []
        for i in range(depth):
            inp = feat_dim if i == 0 else hidden
            layers.append(nn.Linear(inp, hidden))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        if self.fourier_freqs > 0:
            x = torch.cat([x, fourier_encode(x, self.fourier_freqs)], dim=-1)
        return self.mlp(x).squeeze(-1)
