import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        # x: (L, 1, 1, 3C)
        cav_num = x.size(0)

        if self.radix > 1:
            # x: (L, 1, 3, C)
            x = x.view(cav_num,
                       self.cardinality, self.radix, -1)
            x = F.softmax(x, dim=2)
            # 3LC
            x = x.reshape(-1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttn(nn.Module):
    def __init__(self, input_dim):
        super(SplitAttn, self).__init__()
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, input_dim, bias=False)
        self.bn1 = nn.LayerNorm(input_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, input_dim * 3, bias=False)

        self.rsoftmax = RadixSoftmax(3, 1)

    def forward(self, window_list: List[torch.Tensor]) -> torch.Tensor:
        # window list: [(L, H, W, C) * 3]
        assert len(window_list) == 3, 'only 3 windows are supported'

        sw, mw, bw = window_list[0], window_list[1], window_list[2]
        L = sw.shape[0]

        # global average pooling, L, H, W, C
        x_gap = sw + mw + bw
        # L, 1, 1, C
        x_gap = x_gap.mean((1, 2), keepdim=True)
        x_gap = self.act1(self.bn1(self.fc1(x_gap)))
        # L, 1, 1, 3C
        x_attn = self.fc2(x_gap)
        # L, 1, 1, 3C
        x_attn = self.rsoftmax(x_attn).view(L, 1, 1, -1)

        out = sw * torch.narrow(x_attn, 3, 0, self.input_dim) + \
              mw * torch.narrow(x_attn, 3, self.input_dim, self.input_dim) +\
              bw * torch.narrow(x_attn, 3, 2*self.input_dim, self.input_dim)

        return out
