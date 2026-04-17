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

        self.num_pixels = self.ny * self.nx
        self.canvas_size = (self.num_bev_features, self.nx * self.ny * self.max_cav)
        
        assert self.nz == 1

    def forward(self, voxel_coords, pillar_features):
        if voxel_coords.numel() == 0:
            raise ValueError('PointPillarScatter received an empty voxel_coords tensor.')

        batch_size = int(voxel_coords[:, 0].max().item()) + 1
        indices = (voxel_coords[:, 0] * self.num_pixels + voxel_coords[:, 2] * self.nx + voxel_coords[:, 3]).long()

        if indices.min().item() < 0 or indices.max().item() >= batch_size * self.num_pixels:
            raise ValueError(
                f'PointPillarScatter index out of range: min={indices.min().item()}, '
                f'max={indices.max().item()}, capacity={batch_size * self.num_pixels}'
            )

        # canvas: [C, B*max_cav*H*W]
        canvas = torch.zeros(
            (self.num_bev_features, batch_size * self.num_pixels),
            dtype=pillar_features.dtype,
            device=pillar_features.device,
        )

        indices_expanded = indices.unsqueeze(0).expand(self.num_bev_features, -1)
        canvas.scatter_(1, indices_expanded, pillar_features.t())

        # reshape and permute to [total_cavs_in_batch, C, H, W]
        batch_spatial_features = canvas.view(self.num_bev_features, batch_size, self.ny, self.nx)
        batch_spatial_features = batch_spatial_features.permute(1, 0, 2, 3).contiguous()

        return batch_spatial_features
