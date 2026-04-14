# -*- coding: utf-8 -*-

import torch

def regroup(dense_feature: torch.Tensor, record_len: torch.Tensor, max_len: int):
    """
    Regroup the data based on the record_len.

    Parameters
    ----------
    dense_feature : torch.Tensor
        N, C, H, W
    record_len : list
        [sample1_len, sample2_len, ...]
    max_len : int
        Maximum cav number

    Returns
    -------
    regroup_feature : torch.Tensor
        B, L, C, H, W
    mask : torch.Tensor
        B, L
    """
    N, C, H, W = dense_feature.shape
    B = record_len.shape[0]
    L = max_len
    device = 'cuda'
    dtype = dense_feature.dtype

    # binary mask (B, L)
    lp_indices = torch.arange(L, device=device).view(1, L).expand(B, L)
    mask = lp_indices < record_len.unsqueeze(1) # [B, L]

    # flatten dense features to [N, V] where V = C*H*W
    dense_flat = dense_feature.view(N, -1)
    V = dense_flat.shape[1]

    regroup_features = dense_flat.view(B, L, V)
    
    regroup_features = regroup_features * mask.unsqueeze(-1).to(dtype)
    regroup_features = regroup_features.view(B, L, C, H, W)

    return regroup_features, mask.to(dtype)
