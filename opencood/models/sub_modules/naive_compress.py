import torch
import torch.nn as nn
from opencood.tools.quantization_utils import QuantizedConv2D, AffineFakeQuantizer

class NaiveCompressor(nn.Module):
    """
    A very naive compression that only compress on the channel.
    """
    def __init__(self, input_dim, model_cfg):
        super().__init__()

        compress_ratio = model_cfg['compression_ratio']
        quantize_cfg = model_cfg['quantize']

        self.encoder = nn.Sequential(
            QuantizedConv2D(input_dim, input_dim//compress_ratio, kernel_size=3,
                      stride=1, padding=1, quantize_cfg=quantize_cfg['encoder']),
            nn.BatchNorm2d(input_dim//compress_ratio, eps=1e-3, momentum=0.01),
            AffineFakeQuantizer(quantize_cfg['encoder']['bn_type']),
            nn.ReLU(),
            AffineFakeQuantizer(quantize_cfg['encoder']['relu_type'])
        )
        self.decoder = nn.Sequential(
            QuantizedConv2D(input_dim//compress_ratio, input_dim, kernel_size=3,
                      stride=1, padding=1, quantize_cfg=quantize_cfg['decoder']['firstPass']),
            nn.BatchNorm2d(input_dim, eps=1e-3, momentum=0.01),
            AffineFakeQuantizer(quantize_cfg['decoder']['firstPass']['bn_type']),
            nn.ReLU(),
            AffineFakeQuantizer(quantize_cfg['decoder']['firstPass']['relu_type']),
            QuantizedConv2D(input_dim, input_dim, kernel_size=3, 
                        stride=1, padding=1, quantize_cfg=quantize_cfg['decoder']['secondPass']),
            nn.BatchNorm2d(input_dim, eps=1e-3, momentum=0.01),
            AffineFakeQuantizer(quantize_cfg['decoder']['secondPass']['bn_type']),
            nn.ReLU(),
            AffineFakeQuantizer(quantize_cfg['decoder']['secondPass']['relu_type'])
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x