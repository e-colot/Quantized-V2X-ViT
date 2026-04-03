import numpy as np
import torch
from einops import rearrange

from opencood.models.fuse_modules.fuse_utils import regroup


def legacy_regroup(dense_feature, record_len, max_len):
    cum_sum_len = list(np.cumsum(record_len.detach().cpu().numpy()))
    split_features = torch.tensor_split(dense_feature, cum_sum_len[:-1])
    regroup_features = []
    mask = []

    for split_feature in split_features:
        feature_shape = split_feature.shape
        padding_len = max_len - feature_shape[0]
        mask.append([1] * feature_shape[0] + [0] * padding_len)

        padding_tensor = torch.zeros(split_feature.shape[0] * 0 + padding_len,
                                     feature_shape[1],
                                     feature_shape[2],
                                     feature_shape[3],
                                     dtype=split_feature.dtype,
                                     device=split_feature.device)
        split_feature = torch.cat([split_feature, padding_tensor], dim=0)
        split_feature = split_feature.view(-1,
                                           feature_shape[2],
                                           feature_shape[3]).unsqueeze(0)
        regroup_features.append(split_feature)

    regroup_features = torch.cat(regroup_features, dim=0)
    regroup_features = rearrange(regroup_features,
                                 'b (l c) h w -> b l c h w',
                                 l=max_len)
    mask = torch.from_numpy(np.array(mask)).to(regroup_features.device)

    return regroup_features, mask


def make_case(record_len, max_len=5, channels=4, height=2, width=3):
    total_agents = int(record_len.sum().item())
    dense_feature = torch.arange(
        total_agents * channels * height * width,
        dtype=torch.float32,
    ).view(total_agents, channels, height, width)
    return dense_feature, record_len, max_len


def assert_same_outputs(case_name, dense_feature, record_len, max_len):
    new_feature, new_mask = regroup(dense_feature, record_len, max_len)
    old_feature, old_mask = legacy_regroup(dense_feature, record_len, max_len)

    torch.testing.assert_close(new_feature, old_feature)
    torch.testing.assert_close(new_mask, old_mask)
    print(f'{case_name}: OK')


def main():
    # Export path case: a single sample with three CAVs.
    dense_feature, record_len, max_len = make_case(torch.tensor([3]))
    assert_same_outputs('single-sample path', dense_feature, record_len, max_len)

    # Legacy fallback case: multiple samples in the batch.
    dense_feature, record_len, max_len = make_case(torch.tensor([2, 3]))
    assert_same_outputs('multi-sample path', dense_feature, record_len, max_len)

    print('regroup smoke test passed')


if __name__ == '__main__':
    main()