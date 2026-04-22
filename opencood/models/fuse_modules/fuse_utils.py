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
        L, C, H, W
    mask : torch.Tensor
        L
    """
    L = max_len
    device = 'cuda'
    dtype = dense_feature.dtype

    # binary mask (L)
    lp_indices = torch.arange(L, device=device, dtype=torch.int32)
    mask = lp_indices < record_len[0] # (L)

    # flatten dense features to [N, V] where V = C*H*W
    dense_flat = dense_feature.view(dense_feature.shape[0], -1)
    V = dense_flat.shape[1]

    regroup_features = dense_flat.view(L, V)
    
    regroup_features = regroup_features * mask.unsqueeze(-1).to(dtype)
    regroup_features = regroup_features.view(L, dense_feature.shape[1], dense_feature.shape[2], dense_feature.shape[3])

    return regroup_features, mask.to(dtype)
