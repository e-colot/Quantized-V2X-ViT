import numpy as np
import torch
import torch.nn as nn

from opencood.models.fuse_modules.self_attn import AttFusion
from opencood.models.sub_modules.auto_encoder import AutoEncoder


class AttBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.compress = False

        if 'compression' in model_cfg and model_cfg['compression'] > 0:
            self.compress = True
            self.compress_layer = model_cfg['compression']

        if 'layer_nums' in self.model_cfg:

            assert len(self.model_cfg['layer_nums']) == \
                   len(self.model_cfg['layer_strides']) == \
                   len(self.model_cfg['num_filters'])

            layer_nums = self.model_cfg['layer_nums']
            layer_strides = self.model_cfg['layer_strides']
            num_filters = self.model_cfg['num_filters']
        else:
            layer_nums = layer_strides = num_filters = []

        if 'upsample_strides' in self.model_cfg:
            assert len(self.model_cfg['upsample_strides']) \
                   == len(self.model_cfg['num_upsample_filter'])

            num_upsample_filters = self.model_cfg['num_upsample_filter']
            upsample_strides = self.model_cfg['upsample_strides']

        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]

        self.blocks = nn.ModuleList()
        self.fuse_modules = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        self.compression_modules = nn.ModuleList()

        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]

            fuse_network = AttFusion(num_filters[idx])
            self.fuse_modules.append(fuse_network)
            if self.compress and self.compress_layer - idx > 0:
                self.compression_modules.append(AutoEncoder(num_filters[idx],
                                                            self.compress_layer-idx))
            else:
                self.compression_modules.append(nn.Identity())

            for _ in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx],
                              kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])

            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            in_channels=num_filters[idx], 
                            out_channels=num_upsample_filters[idx],
                            kernel_size=upsample_strides[idx],
                            stride=upsample_strides[idx], 
                            bias=False,
                            output_padding=0
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx],
                                       eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = int(np.round(1 / stride))
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            kernel_size=stride,
                            stride=stride, 
                            bias=False,
                            output_padding=0
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3,
                                       momentum=0.01),
                        nn.ReLU()
                    ))
            else:
                self.deblocks.append(nn.Identity())

        self.upsample_stride_len = len(upsample_strides)
        self.num_bev_features = sum(num_upsample_filters)

        self.extra_deblock = nn.Identity()
        if self.upsample_stride_len > num_levels:
            self.extra_deblock = nn.Sequential(
                nn.ConvTranspose2d(self.num_bev_features, self.num_bev_features, upsample_strides[-1], 
                                        stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(self.num_bev_features, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )

    def forward(self, batch_spatial_features, record_len):

        ups = []
        x = batch_spatial_features
            
        for block, compression_module, fuse_module, deblock in zip(
            self.blocks, self.compression_modules, self.fuse_modules, self.deblocks):
            x = block(x)
            x = compression_module(x)
            x = fuse_module(x, record_len)

            ups.append(deblock(x))

        if self.upsample_stride_len > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]

        x = self.extra_deblock(x)

        return x
