import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg['num_features']
        self.max_cav = self.model_cfg.get('max_cav', None)
        
        grid_size = model_cfg['grid_size']
        self.nx = torch.tensor(grid_size[0], dtype=torch.int32, device='cuda')
        self.ny = torch.tensor(grid_size[1], dtype=torch.int32, device='cuda')
        self.nz = torch.tensor(grid_size[2], dtype=torch.int32, device='cuda')

        self.num_pixels = self.ny * self.nx
        self.canvas_second_dim = self.num_pixels * self.max_cav
        
        assert self.nz == 1

    def forward(self, voxel_coords, pillar_features):
        if voxel_coords.numel() == 0:
            raise ValueError('PointPillarScatter received an empty voxel_coords tensor.')

        indices = (torch.select(voxel_coords, 1, 0).to(torch.int32) * self.num_pixels +
                torch.select(voxel_coords, 1, 2).to(torch.int32) * self.nx +
                torch.select(voxel_coords, 1, 3).to(torch.int32))

        # if indices.min().item() < 0 or indices.max().item() >= batch_size * self.num_pixels:
        #     raise ValueError(
        #         f'PointPillarScatter index out of range: min={indices.min().item()}, '
        #         f'max={indices.max().item()}, capacity={batch_size * self.num_pixels}'
        #     )

        # canvas: [C, B*max_cav*H*W]
        canvas = torch.zeros(
            (self.num_bev_features, self.canvas_second_dim),
            dtype=pillar_features.dtype,
            device=pillar_features.device,
        )

        indices_expanded = indices.unsqueeze(0).expand(self.num_bev_features, -1)
        canvas.scatter_(1, indices_expanded, pillar_features.t())

        # reshape and permute to [total_cavs_in_batch, C, H, W]
        batch_spatial_features = canvas.view(self.num_bev_features, self.max_cav, self.ny, self.nx)
        batch_spatial_features = batch_spatial_features.permute(1, 0, 2, 3).contiguous()

        return batch_spatial_features
