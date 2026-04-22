"""
Pillar VFE, credits to OpenPCDet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Wrapper to get rid of a Long node in graph
@torch.jit.script
def trt_max(x: torch.Tensor):
    val, _ = torch.max(x, dim=1, keepdim=True)
    return val

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):

        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x

        x = F.relu(x)
        x_max = trt_max(x)

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.expand(-1, inputs.shape[1], x.shape[-1])
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(nn.Module):
    def __init__(self, model_cfg, num_point_features, voxel_size,
                 point_cloud_range):
        super().__init__()
        self.model_cfg = model_cfg

        self.use_norm = self.model_cfg['use_norm']
        self.with_distance = self.model_cfg['with_distance']

        self.use_absolute_xyz = self.model_cfg['use_absolute_xyz']
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg['num_filters']
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm,
                         last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.register_buffer('voxel_x', torch.tensor(voxel_size[0], dtype=torch.float32))
        self.register_buffer('voxel_y', torch.tensor(voxel_size[1], dtype=torch.float32))
        self.register_buffer('voxel_z', torch.tensor(voxel_size[2], dtype=torch.float32))

        self.register_buffer('x_offset', torch.tensor(voxel_size[0]/2 + point_cloud_range[0], dtype=torch.float32))
        self.register_buffer('y_offset', torch.tensor(voxel_size[1]/2 + point_cloud_range[1], dtype=torch.float32))
        self.register_buffer('z_offset', torch.tensor(voxel_size[2]/2 + point_cloud_range[2], dtype=torch.float32))

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    @staticmethod
    def get_paddings_indicator(actual_num: torch.Tensor, max_num: int, axis: int = 0):
        # actual_num: [N]  →  unsqueeze to [N, 1]
        actual_num = torch.unsqueeze(actual_num, axis + 1)   # [N, 1]
        # arange: [max_num] → unsqueeze to [1, max_num]
        # Explicit shape instead of view([1,-1]) which TRT rejects
        max_num_t = torch.arange(max_num, dtype=torch.int32, device=actual_num.device).unsqueeze(0)  # [1, max_num]
        paddings_indicator = actual_num.to(torch.int32) > max_num_t    # [N, max_num] broadcast
        return paddings_indicator
    
    def forward(self, voxel_features, voxel_coords, voxel_num_points):

        points_mean = \
            torch.narrow(voxel_features, 2, 0, 3).sum(dim=1, keepdim=True) / \
            voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = torch.narrow(voxel_features, 2, 0, 3) - points_mean

        voxel_coords = voxel_coords.to(voxel_features.dtype)
        
        row0 = torch.select(voxel_features, 2, 0) - (torch.select(voxel_coords, 1, 3).unsqueeze(1) * self.voxel_x + self.x_offset)
        row1 = torch.select(voxel_features, 2, 1) - (torch.select(voxel_coords, 1, 2).unsqueeze(1) * self.voxel_y + self.y_offset)
        row2 = torch.select(voxel_features, 2, 2) - (torch.select(voxel_coords, 1, 1).unsqueeze(1) * self.voxel_z + self.z_offset)
        
        f_center = torch.stack((row0, row1, row2), dim=2)

        kept_features = voxel_features \
            if self.use_absolute_xyz \
            else torch.narrow(voxel_features, 2, 3, voxel_features.shape[2]-3)

        if self.with_distance:
            points_dist = torch.norm(torch.narrow(voxel_features, 2, 0, 3), 2, 2, keepdim=True)
            features = torch.cat((kept_features, f_cluster, f_center, points_dist), dim=-1)
        else:
            features = torch.cat((kept_features, f_cluster, f_center), dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count,
                                           axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
    
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze(1)
        return features
