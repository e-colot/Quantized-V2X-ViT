"""
Class used to downsample features by 3*3 conv
"""
import torch.nn as nn
from opencood.tools.quantization_utils import QuantizedConv2D, AffineFakeQuantizer


class DoubleConv(nn.Module):
    """
    Double convoltuion
    Args:
        in_channels: input channel num
        out_channels: output channel num
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, quantize_cfg):
        super().__init__()

        relu1_d_type = 'fp32'
        if quantize_cfg['relu1']['quantize_relu']:
            relu1_d_type = quantize_cfg['relu1']['type']
        relu2_d_type = 'fp32'
        if quantize_cfg['relu2']['quantize_relu']:
            relu2_d_type = quantize_cfg['relu2']['type']
        
        self.double_conv = nn.Sequential(
            QuantizedConv2D(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, quantize_cfg=quantize_cfg['conv1']),
            nn.ReLU(inplace=True),
            AffineFakeQuantizer(relu1_d_type),
            QuantizedConv2D(out_channels, out_channels, kernel_size=3, padding=1, 
                            quantize_cfg=quantize_cfg['conv2']),
            nn.ReLU(inplace=True),
            AffineFakeQuantizer(relu2_d_type)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownsampleConv(nn.Module):
    def __init__(self, config):
        super(DownsampleConv, self).__init__()
        self.layers = nn.ModuleList([])
        input_dim = config['input_dim']

        for (ksize, dim, stride, padding, quantize_cfg) in zip(config['kernal_size'], # haha, typo loser
                                                 config['dim'],
                                                 config['stride'],
                                                 config['padding'],
                                                 config['quantize']):
            self.layers.append(DoubleConv(input_dim,
                                          dim,
                                          kernel_size=ksize,
                                          stride=stride,
                                          padding=padding,
                                          quantize_cfg=quantize_cfg))
            input_dim = dim

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x