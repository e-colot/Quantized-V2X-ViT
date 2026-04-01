import numpy as np
import torch
import torch.nn as nn
from opencood.tools.quantization_utils import QuantizedConv2D, QuantizedConvTranspose2D, AffineFakeQuantizer


class BaseBEVBackbone(nn.Module):
    # Note that there is currently no support for different data types in a single layer
    # Current model has layer_nums = [3, 5, 8] so the quantization level is the same for the first 3 layers, 
    # for the 5 intermediate layers and for the last 8 layers
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        self.quantize_cfg = self.model_cfg.get('quantize', {})

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
        self.deblocks = nn.ModuleList()

        for idx in range(num_levels):
            layer_quantize_cfg = self.quantize_cfg['layers'][idx]
            upsample_quantize_cfg = self.quantize_cfg['upsample'][idx]

            batchnorm_d_type = 'fp32'
            relu_d_type = 'fp32'
            if layer_quantize_cfg['batchnorm']['quantize_batchnorm']:
                batchnorm_d_type = layer_quantize_cfg['batchnorm']['type']
            if layer_quantize_cfg['relu']['quantize_relu']:
                relu_d_type = layer_quantize_cfg['relu']['type']

            cur_layers = [
                nn.ZeroPad2d(1),
                QuantizedConv2D(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False, quantize_cfg=layer_quantize_cfg['conv']
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                AffineFakeQuantizer(batchnorm_d_type),
                nn.ReLU(),
                AffineFakeQuantizer(relu_d_type)
            ]

            for _ in range(layer_nums[idx]):
                cur_layers.extend([
                    QuantizedConv2D(num_filters[idx], num_filters[idx],
                              kernel_size=3, padding=1, bias=False, quantize_cfg=layer_quantize_cfg['conv']),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    AffineFakeQuantizer(batchnorm_d_type),
                    nn.ReLU(),
                    AffineFakeQuantizer(relu_d_type)
                ])

            self.blocks.append(nn.Sequential(*cur_layers))

            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]

                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        QuantizedConvTranspose2D(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False, quantize_cfg=upsample_quantize_cfg['convTranspose']
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx],
                                       eps=1e-3, momentum=0.01),
                        AffineFakeQuantizer(batchnorm_d_type),
                        nn.ReLU(),
                        AffineFakeQuantizer(relu_d_type)
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        QuantizedConv2D(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False, quantize_cfg=upsample_quantize_cfg['convTranspose']
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3,
                                       momentum=0.01),
                        AffineFakeQuantizer(batchnorm_d_type),
                        nn.ReLU(),
                        AffineFakeQuantizer(relu_d_type)
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            upsample_quantize_cfg = self.quantize_cfg['upsample'][-1]

            batchnorm_d_type = 'fp32'
            relu_d_type = 'fp32'
            if upsample_quantize_cfg['batchnorm']['quantize_batchnorm']:
                batchnorm_d_type = upsample_quantize_cfg['batchnorm']['type']
            if upsample_quantize_cfg['relu']['quantize_relu']:
                relu_d_type = upsample_quantize_cfg['relu']['type']

            self.deblocks.append(nn.Sequential(
                QuantizedConvTranspose2D(c_in, c_in, upsample_strides[-1],
                                   stride=upsample_strides[-1], bias=False, quantize_cfg=upsample_quantize_cfg['convTranspose']),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                AffineFakeQuantizer(batchnorm_d_type),
                nn.ReLU(),
                AffineFakeQuantizer(relu_d_type)
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']

        ups = []
        ret_dict = {}
        x = spatial_features

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x
        return data_dict
