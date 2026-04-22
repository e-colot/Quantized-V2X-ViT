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
        # batch_size = 1 during inference
        if prior_encoding.shape[0] != 1 or spatial_correction_matrix.shape[0] != 1:
            raise NotImplementedError('Model has been restricted to a batch size of 1 for inference purposes')
        prior_encoding = prior_encoding.squeeze(0)
        spatial_correction_matrix = spatial_correction_matrix.squeeze(0)

        pillar_features = self.pillar_vfe(voxel_features, voxel_coords, voxel_num_points)
        # (N, 4) -> (N, C)

        spatial_features = self.scatter(voxel_coords, pillar_features)
        # (N, C) -> (N, C, H, W)

        spatial_features_2d = self.backbone(spatial_features)

        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)


        regroup_feature, mask = regroup(spatial_features_2d,
                                        record_len,
                                        self.max_cav)
        # (N, C, H, W) -> (L, C, H, W)

        # prior encoding added
        prior_encoding = prior_encoding.repeat(1, 1,
                                               regroup_feature.shape[2],
                                               regroup_feature.shape[3])
        regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=1)

        regroup_feature = regroup_feature.permute(0, 2, 3, 1)
        # (L, C, H, W) -> (L, H, W, C)

        # transformer fusion
        fused_feature = self.fusion_net(regroup_feature, mask, spatial_correction_matrix)
        fused_feature = fused_feature.permute(2, 0, 1)
        # (H, W, C) -> (C, H, W)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        return psm, rm
