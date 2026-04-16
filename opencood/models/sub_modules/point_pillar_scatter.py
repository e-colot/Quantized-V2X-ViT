import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        device = 'cuda'

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg['num_features']
        self.max_cav = self.model_cfg.get('max_cav', None)
        
        grid_size = model_cfg['grid_size']
        self.nx = torch.tensor([grid_size[0]], dtype=torch.int32, device=device)
        self.ny = torch.tensor([grid_size[1]], dtype=torch.int32, device=device)
        self.nz = torch.tensor([grid_size[2]], dtype=torch.int32, device=device)

        self.num_pixels = self.ny * self.nx
        
        assert self.nz == 1

    def forward(self, voxel_coords, pillar_features):
        indices = (voxel_coords[:, 0] * self.num_pixels + voxel_coords[:, 2] * self.nx + voxel_coords[:, 3])

        # canvas: [C, B*max_cav*H*W]
        canvas = pillar_features.new_zeros((self.num_bev_features, self.nx * self.ny * self.max_cav))

        indices_expanded = indices.unsqueeze(0).expand(self.num_bev_features, -1)
        canvas.scatter_(1, indices_expanded, pillar_features.t())

        # reshape and permute to [B*max_cav, C, H, W]
        batch_spatial_features = canvas.view(self.num_bev_features, self.max_cav, self.ny, self.nx)
        batch_spatial_features = batch_spatial_features.permute(1, 0, 2, 3).contiguous()

        return batch_spatial_features
