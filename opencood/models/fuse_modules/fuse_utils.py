# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import numpy as np

from einops import rearrange
from opencood.utils.common_utils import torch_tensor_to_numpy


def regroup(dense_feature, record_len, max_len):
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
    """
    if record_len.shape[0] == 1:
        feature_shape = dense_feature.shape
        padding_len = max_len - feature_shape[0]

        padding_tensor = dense_feature.new_zeros(padding_len,
                                                 feature_shape[1],
                                                 feature_shape[2],
                                                 feature_shape[3])
        split_feature = torch.cat([dense_feature, padding_tensor], dim=0)
        regroup_features = split_feature.view(1,
                                              -1,
                                              feature_shape[2],
                                              feature_shape[3])
        regroup_features = rearrange(regroup_features,
                                     'b (l c) h w -> b l c h w',
                                     l=max_len)
        mask = torch.cat([
            torch.ones(feature_shape[0], device=dense_feature.device,
                       dtype=torch.long),
            torch.zeros(padding_len, device=dense_feature.device,
                        dtype=torch.long)
        ], dim=0).unsqueeze(0)
        return regroup_features, mask

    cum_sum_len = list(np.cumsum(torch_tensor_to_numpy(record_len)))
    split_features = torch.tensor_split(dense_feature,
                                        cum_sum_len[:-1])
    regroup_features = []
    mask = []

    for split_feature in split_features:
        # M, C, H, W
        feature_shape = split_feature.shape

        # the maximum M is 5 as most 5 cavs
        padding_len = max_len - feature_shape[0]
        mask.append([1] * feature_shape[0] + [0] * padding_len)

        padding_tensor = torch.zeros(padding_len, feature_shape[1],
                                     feature_shape[2], feature_shape[3])
        padding_tensor = padding_tensor.to(split_feature.device)

        split_feature = torch.cat([split_feature, padding_tensor],
                                  dim=0)

        # 1, 5C, H, W
        split_feature = split_feature.view(-1,
                                           feature_shape[2],
                                           feature_shape[3]).unsqueeze(0)
        regroup_features.append(split_feature)

    # B, 5C, H, W
    regroup_features = torch.cat(regroup_features, dim=0)
    # B, L, C, H, W
    regroup_features = rearrange(regroup_features,
                                 'b (l c) h w -> b l c h w',
                                 l=max_len)
    mask = torch.from_numpy(np.array(mask)).to(regroup_features.device)

    return regroup_features, mask
