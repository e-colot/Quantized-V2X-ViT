import math

from opencood.models.sub_modules.base_transformer import *
from opencood.models.fuse_modules.hmsa import *
from opencood.models.fuse_modules.mswin import *
from opencood.models.sub_modules.torch_transformation_utils import \
    get_transformation_matrix, warp_affine, get_roi_and_cav_mask, \
    get_discretized_transformation_matrix, clamp_tensor
from torch import nn


import torch
from torch import nn

class STTF(nn.Module):
    def __init__(self, args):
        super(STTF, self).__init__()
        device = 'cuda'
        # Register these as buffers to keep them as Tensors in the graph
        self.register_buffer('discrete_ratio', torch.tensor(args['voxel_size'][0], dtype=torch.float32, device=device))
        self.register_buffer('downsample_rate', torch.tensor(float(args['downsample_rate']), dtype=torch.float32, device=device))

    def forward(self, x, spatial_correction_matrix):
        r"""
        x: (L, H, W, C)
        spatial_correction_matrix: (M, 4, 4)
        """
        device = 'cuda'
        orig_shape = x.shape # (L, H, W, C)

        # x shape: (L, H, W, C) -> (L, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # spatial size = [H, W]
        spatial_size = torch.tensor([orig_shape[1], orig_shape[2]], device=device, dtype=torch.int32)

        # dist_correction_matrix: (M, 2, 3)
        dist_correction_matrix = get_discretized_transformation_matrix(
            spatial_correction_matrix, 
            self.discrete_ratio,
            self.downsample_rate
        )

        # transformation_matrix: (M, 2, 3)
        transformation_matrix = get_transformation_matrix(dist_correction_matrix, spatial_size)
        
        # x_warped: (L, C, H, W)
        x_warped = warp_affine(x, transformation_matrix, spatial_size)
        
        # (L, C, H, W) -> (L, H, W, C)
        x = x_warped.permute(0, 2, 3, 1).contiguous()
        
        return x


class RTE(nn.Module):
    def __init__(self, dim, RTE_ratio=2):
        super(RTE, self).__init__()
        self.RTE_ratio = RTE_ratio
        self.emb = RelTemporalEncoding(dim, RTE_ratio=self.RTE_ratio)

    def forward(self, x, dts):
        r"""
        x: (L, H, W, C)
        dts: (L)
        """
        
        # Result: (L, H, W, C)
        x_rte = self.emb(x, dts)
        
        return x_rte


class RelTemporalEncoding(nn.Module):
    def __init__(self, n_hid, RTE_ratio, max_len=100):
        super(RelTemporalEncoding, self).__init__()
        # Use a buffer so it's moved to GPU automatically
        div_term = torch.exp(torch.arange(0, n_hid, 2, dtype=torch.float32) * -(math.log(10000.0) / n_hid))
        
        # Use an actual Embedding layer or a functional approach
        self.emb = nn.Embedding(max_len, n_hid)
        
        # Precompute sinusoid weights
        position = torch.arange(0., max_len, dtype=torch.float32).unsqueeze(1)
        self.emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        self.emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        self.emb.weight.requires_grad = False
        
        self.register_buffer('RTE_ratio', torch.tensor(RTE_ratio, dtype=torch.int32, device='cuda'))
        self.lin = nn.Linear(n_hid, n_hid)

        self.num_embeddings = torch.tensor(self.emb.num_embeddings, dtype=torch.int32, device='cuda')
        self.tensor_zero = torch.tensor(0, dtype=torch.int32, device='cuda')

    def forward(self, x, t):
        r"""
        x: (L, H, W, C)
        t: (L)
        """
        
        # Clamp is a safety for tensorRT conversion (required)
        indices = clamp_tensor(t * self.RTE_ratio, self.tensor_zero, self.num_embeddings)

        # Get temporal embeddings: (L, C)
        t_emb = self.lin(self.emb(indices))
        
        # Broadcast across H and W: (L, 1, 1, C)
        t_emb = t_emb.unsqueeze(1).unsqueeze(2)
        
        # combined feature
        return x + t_emb


class V2XFusionBlock(nn.Module):
    def __init__(self, num_blocks, cav_att_config, pwindow_config):
        super().__init__()
        # first multi-agent attention and then multi-window attention
        self.layers = nn.ModuleList([])
        self.num_blocks = num_blocks

        for _ in range(num_blocks):
            att = PreNormedHGTCavAttention(cav_att_config['dim'],
                                  heads=cav_att_config['heads'],
                                  dim_head=cav_att_config['dim_head'],
                                  dropout=cav_att_config['dropout']) if \
                cav_att_config['use_hetero'] else \
                PreNormedCavAttention(cav_att_config['dim'],
                             heads=cav_att_config['heads'],
                             dim_head=cav_att_config['dim_head'],
                             dropout=cav_att_config['dropout'])
            self.layers.append(nn.ModuleList([
                att,
                PreNormedPyramidWindowAttention(cav_att_config['dim'],
                                        pwindow_config['dim'],
                                        heads=pwindow_config['heads'],
                                        dim_heads=pwindow_config[
                                            'dim_head'],
                                        drop_out=pwindow_config[
                                            'dropout'],
                                        window_size=pwindow_config[
                                            'window_size'],
                                        relative_pos_embedding=
                                        pwindow_config[
                                            'relative_pos_embedding'],
                                        fuse_method=pwindow_config[
                                            'fusion_method'])]))

    def forward(self, x, mask, prior_encoding):
        r"""
        x: (L, H, W, C)
        mask: (H, W, 1, L)
        prior_encoding: (L, H, W, 3)
        """
        for layer in self.layers:
            x = layer[0](x, mask=mask, prior_encoding=prior_encoding) + x
            # x: (L, H, W, C)
            x = layer[1](x) + x
            # x: (L, H, W, C)
        return x


class V2XTEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        cav_att_config = args['cav_att_config']
        pwindow_att_config = args['pwindow_att_config']
        feed_config = args['feed_forward']

        num_blocks = args['num_blocks']
        depth = args['depth']
        mlp_dim = feed_config['mlp_dim']
        dropout = feed_config['dropout']

        self.downsample_rate = torch.tensor(args['sttf']['downsample_rate'], dtype=torch.float32, device='cuda')
        self.discrete_ratio = torch.tensor(args['sttf']['voxel_size'][0], dtype=torch.float32, device='cuda')
        self.use_roi_mask = args['use_roi_mask']
        self.use_RTE = cav_att_config['use_RTE']
        self.RTE_ratio = cav_att_config['RTE_ratio']
        self.sttf = STTF(args['sttf'])
        # adjust the channel numbers from 256+3 -> 256
        self.prior_feed = nn.Linear(cav_att_config['dim'] + 3,
                                    cav_att_config['dim'])
        self.layers = nn.ModuleList([])
        if self.use_RTE:
            self.rte = RTE(cav_att_config['dim'], self.RTE_ratio)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                V2XFusionBlock(num_blocks, cav_att_config, pwindow_att_config),
                PreNormedFeedForward(cav_att_config['dim'], mlp_dim,
                                    dropout=dropout)]))

    def forward(self, x, mask, spatial_correction_matrix):
        r"""
        x: (L, H, W, C)
        mask: (L)
        spatial_correction_matrix: (M, 4, 4)
        """

        C = int(x.shape[-1])

        # transform the features to the current timestamp
        # velocity, time_delay, infra
        # prior_encoding: (L, H, W, 3)
        prior_encoding = torch.narrow(x, 3, C-3, 3)

        # x: (L, H, W, C)
        x = torch.narrow(x, 3, 0, C-3)

        if self.use_RTE:
            # dt: (L)
            dt = torch.select(torch.select(torch.select(prior_encoding, 3, 1), 2, 0), 1, 0).to(torch.int32)
            x = self.rte(x, dt)

        # x: (L, H, W, C)
        x = self.sttf(x, spatial_correction_matrix)

        # probably throwing an error if not self.use_roi_mask -> mask: (L, 1, 1, 1)
        # mask: (H, W, 1, L)
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(
            3) if not self.use_roi_mask else get_roi_and_cav_mask(x,
                                                                  mask,
                                                                  spatial_correction_matrix,
                                                                  self.discrete_ratio,
                                                                  self.downsample_rate)
        for layer in self.layers:
            x = layer[0](x, mask=com_mask, prior_encoding=prior_encoding)
            # x: (L, H, W, C)
            x = layer[1](x) + x
        return x


class V2XTransformer(nn.Module):
    def __init__(self, args):
        super(V2XTransformer, self).__init__()

        encoder_args = args['encoder']
        self.encoder = V2XTEncoder(encoder_args)

    def forward(self, x, mask, spatial_correction_matrix):
        r"""
        x: (L, H, W, C)
        mask: (L)
        spatial_correction_matrix: (M, 4, 4) where M is the max_cav
        """
        output = self.encoder(x, mask, spatial_correction_matrix)
        # output: (L, H, W, C)

        output = torch.select(output, 0, 0)
        # output: (H, W, C)
        return output
