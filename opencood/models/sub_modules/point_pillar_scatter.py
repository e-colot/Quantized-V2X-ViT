import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg['num_features']
        self.max_cav = self.model_cfg.get('max_cav', None)
        
        grid_size = model_cfg['grid_size']
        self.nx = int(grid_size[0])
        self.ny = int(grid_size[1])
        self.nz = int(grid_size[2])
        
        assert self.nz == 1

    def forward(self, voxel_coords, record_len, pillar_features):

        device = 'cuda'
        dtype = pillar_features.dtype

        # In export mode we avoid tensor->python scalar conversions by using
        # a fixed upper bound: num_samples_in_batch * max_cav.
        if self.max_cav is not None:
            upper_batch_bound = record_len.shape[0] * self.max_cav
        else:
            raise NotImplementedError
            # upper_batch_bound = voxel_coords[:, 0].max().int().item() + 1

        # Each pillar is mapped to a unique position in a flattened (B * H * W) grid.
        # voxel_coords[:, 0] is batch_idx,
        # voxel_coords[:, 2] is y,
        # voxel_coords[:, 3] is x.
        # (nz is 1, so voxel_coords[:, 1] is ignored)
        spatial_indices = (voxel_coords[:, 0].to(torch.int64) * (self.nx * self.ny) +
                          voxel_coords[:, 2].to(torch.int64) * self.nx +
                          voxel_coords[:, 3].to(torch.int64))

        batch_spatial_features = torch.zeros((upper_batch_bound, self.num_bev_features, self.nx * self.ny),
            dtype=dtype, device=device)
        
        batch_spatial_features = batch_spatial_features.view(self.num_bev_features, -1)

        full_indices = spatial_indices.unsqueeze(0).expand(self.num_bev_features, -1)
        pillars = pillar_features.t()

        batch_spatial_features.scatter_(1, full_indices, pillars)
        batch_spatial_features = batch_spatial_features.view(
            self.num_bev_features, upper_batch_bound, self.ny, self.nx
        ).permute(1, 0, 2, 3).contiguous()

        return batch_spatial_features
