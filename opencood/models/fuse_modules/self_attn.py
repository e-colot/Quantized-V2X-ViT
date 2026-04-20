import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value, mask=None):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        if mask is not None:
            score = score.masked_fill(mask, float('-inf'))
        attn = F.softmax(score, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        context = torch.bmm(attn, value)
        return context


class AttFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x, record_len):
        """
        x          : [max_cav, C, H, W]
        record_len : int (Python), number of valid CAVs
        Returns    : [1, C, H, W]
        """
        max_cav, C, H, W = x.shape

        tokens = x.view(max_cav, C, -1).permute(2, 0, 1)  # [H*W, max_cav, C]

        # Functional mask — no in-place ops, TRT-safe
        # pad_mask[b, 0, i] = True means slot i is padding (ignore it)
        indices = torch.arange(max_cav, device=x.device)   # [max_cav]
        pad_mask = (indices >= record_len)                  # [max_cav], True=padding
        pad_mask = pad_mask.unsqueeze(0).unsqueeze(0)       # [1, 1, max_cav]
        pad_mask = pad_mask.expand(H * W, 1, max_cav)      # [H*W, 1, max_cav]

        h = self.att(tokens, tokens, tokens, mask=pad_mask) # [H*W, max_cav, C]

        ego = h[:, 0, :]                         # [H*W, C]
        ego = ego.permute(1, 0).view(C, H, W)   # [C, H, W]
        return ego.unsqueeze(0)                  # [1, C, H, W]
    
