# -*- coding: utf-8 -*-

import torch
from typing import List

def regroup(dense_feature: torch.Tensor, record_len: List[int], max_len: int):
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
    
    if not isinstance(record_len, torch.Tensor):
        record_len = torch.tensor(record_len, device=dense_feature.device)
    
    B = record_len.shape[0]
    L = max_len
    device = dense_feature.device
    dtype = dense_feature.dtype

    # Create the grid mask to identify valid slots in the BxL output
    lp_indices = torch.arange(L, device=device).view(1, L).expand(B, L)
    mask = lp_indices < record_len.unsqueeze(1) # Boolean (B, L)

    # Use cumsum to assign a unique sequential index to each True value in the mask
    cumulative_idx = torch.cumsum(mask.view(-1).to(torch.int32), dim=0).view(B, L)
    # Convert to 0-based indexing for the N dimension
    target_n_idx = (cumulative_idx - 1)
    
    # Create a global range of indices for the input N features
    n_range = torch.arange(N, device=device).view(N, 1, 1)
    
    # Generate an assignment matrix: True if the N-th feature belongs in slot (b, l)
    assignment_mask = (n_range == target_n_idx.unsqueeze(0))
    # Ensure only valid mask slots are considered to avoid mapping padding to features
    assignment_mask = assignment_mask * mask.unsqueeze(0)

    # Flatten spatial and channel dims to treat each object as a single vector
    dense_flat = dense_feature.view(N, -1)
    
    # Map features to the grid via Einstein summation (dot product of assignment and features)
    # n,v: N, CHW | n,b,l: N, B, L -> b,l,v: B, L, CHW
    out_flat = torch.einsum('nv,nbl->blv', dense_flat, assignment_mask.to(dtype))
    
    # Reshape the flattened result back into the structured 5D output
    regroup_features = out_flat.view(B, L, C, H, W)

    return regroup_features, mask.to(dtype)