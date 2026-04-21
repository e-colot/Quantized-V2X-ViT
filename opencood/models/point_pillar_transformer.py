import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.fuse_modules.fuse_utils import regroup
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.v2xvit_basic import V2XTransformer


class PointPillarTransformer(nn.Module):
    def __init__(self, args):
        super(PointPillarTransformer, self).__init__()

        self.max_cav = args['max_cav']
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        scatter_cfg = dict(args['point_pillar_scatter'])
        scatter_cfg.setdefault('max_cav', self.max_cav)
        self.scatter = PointPillarScatter(scatter_cfg)
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.fusion_net = V2XTransformer(args['transformer'])

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, voxel_features, voxel_coords, voxel_num_points, record_len, 
                spatial_correction_matrix, prior_encoding):

        # B, max_cav, 3(dt dv infra), 1, 1
        prior_encoding = prior_encoding.unsqueeze(-1).unsqueeze(-1)
        # Relative L2 summary: mean=0.000000e+00, median=0.000000e+00, max=0.000000e+00

        # n, 4 -> n, c
        pillar_features = self.pillar_vfe(voxel_features, voxel_coords, voxel_num_points)
        # Relative L2 summary: mean=2.909690e-04, median=2.876608e-04, max=3.430791e-04

        # n, c -> N, C, H, W
        spatial_features = self.scatter(voxel_coords, pillar_features)
        # Relative L2 summary: mean=2.909690e-04, median=2.876609e-04, max=3.430791e-04

        spatial_features_2d = self.backbone(spatial_features)
        # Relative L2 summary: mean=1.254411e-03, median=1.217855e-03, max=1.686136e-03

        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # Relative L2 summary: mean=8.642389e-04, median=8.403341e-04, max=1.088899e-03

        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        # Relative L2 summary: mean=1.209472e-03, median=1.136962e-03, max=2.426849e-03

        # N, C, H, W -> B,  L, C, H, W

        regroup_feature, mask = regroup(spatial_features_2d,
                                        record_len,
                                        self.max_cav)
        # prior encoding added
        prior_encoding = prior_encoding.repeat(1, 1, 1,
                                               regroup_feature.shape[3],
                                               regroup_feature.shape[4])
        regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)

        # b l c h w -> b l h w c
        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
        # transformer fusion
        fused_feature = self.fusion_net(regroup_feature, mask, spatial_correction_matrix)
        # b h w c -> b c h w
        fused_feature = fused_feature.permute(0, 3, 1, 2)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        return psm, rm
