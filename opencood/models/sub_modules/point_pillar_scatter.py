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
        device = pillar_features.device
        dtype = pillar_features.dtype

        batch_size = record_len.shape[0]
        upper_batch_bound = batch_size * self.max_cav
        
        # Flattened spatial size
        num_pixels = self.ny * self.nx
        total_elements = upper_batch_bound * num_pixels

        # flat indices for the (B*max_cav * H * W) dimension
        indices = (voxel_coords[:, 0].long() * num_pixels +
                   voxel_coords[:, 2].long() * self.nx +
                   voxel_coords[:, 3].long())

        # canvas: [C, B*max_cav*H*W]
        canvas = torch.zeros((self.num_bev_features, total_elements),
                             dtype=dtype, device=device)

        indices_expanded = indices.unsqueeze(0).expand(self.num_bev_features, -1)
        canvas.scatter_(1, indices_expanded, pillar_features.t())

        # reshape and permute to [B*max_cav, C, H, W]
        batch_spatial_features = canvas.view(self.num_bev_features, upper_batch_bound, self.ny, self.nx)
        batch_spatial_features = batch_spatial_features.permute(1, 0, 2, 3).contiguous()

        return batch_spatial_features
