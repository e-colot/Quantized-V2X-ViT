"""
Multi-scale window transformer
"""
import torch
import torch.nn as nn
import numpy as np

from opencood.models.sub_modules.split_attn import SplitAttn


def get_relative_distances(window_size):
    indices = torch.tensor(np.array(
        [[x, y] for x in range(window_size) for y in range(window_size)]), dtype=torch.int32)
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class BaseWindowAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, drop_out, window_size,
                 relative_pos_embedding):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.inner_dim = inner_dim
        self.head_dim = dim_head
        self.register_buffer('heads_tensor', torch.tensor(heads, dtype=torch.int32, device='cuda'))
        self.register_buffer('scale', torch.tensor(dim_head ** -0.5, dtype=torch.float32, device='cuda'))
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:            
            stride = 2 * window_size - 1
            rel_coords = get_relative_distances(window_size) + window_size - 1
            indices_1d = rel_coords[:, :, 0] * stride + rel_coords[:, :, 1]
            indices_1d = indices_1d.clamp(min=0)
            
            # Register the INDEX MAP as a buffer (TensorRT constant)
            self.register_buffer('rel_idx_1d', indices_1d.reshape(-1).to(torch.int32))
            
            # This matches your trained checkpoint shape
            self.pos_embedding = nn.Parameter(torch.randn(stride, stride))
            self.pos_enc_shape = torch.tensor([self.window_size**2, self.window_size**2], dtype=torch.int32, device='cuda')
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2,
                                                          window_size ** 2))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(drop_out)
        )

    def forward(self, x):
        # x shape: (b, l, h, w, c)
        b, l, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
        ws = self.window_size

        qkv = self.to_qkv(x)
        # Prefer explicit split sizes over chunk to avoid zero-length partitions
        # in TensorRT shape propagation when dimensions are symbolic.
        q, k, v = torch.split(qkv, self.inner_dim, dim=-1)
        # all above are (b, l, h, w, c)

        new_h = h // ws
        new_w = w // ws
        qkv_dim = self.inner_dim
        head_dim = self.head_dim

        if new_h < 1: new_h = 1
        if new_w < 1: new_w = 1
     
        # (b, l, h, w, c) -> (b, l, new_h, w_size, w, c) -> (b, l, new_h, w_size, new_w, w_size, c)
        q = q.reshape(b, l, new_h, ws, new_w, ws, qkv_dim)
        # (b, l, new_h, w_size, new_w, w_size, c) -> (b, l, new_h, w_size, new_w, w_size, heads, c_heads)
        q = q.reshape(b, l, new_h, ws, new_w, ws, self.heads, head_dim)
        # (b, l, new_h, w_size, new_w, w_size, heads, c_heads) -> (b, l, heads, new_h, new_w, w_size, w_size, c_heads)
        q = q.permute(0, 1, 6, 2, 4, 3, 5, 7).contiguous()
        # (b, l, heads, new_h, new_w, w_size, w_size, c_heads) -> (b, l, heads, new_h*new_w, w_size*w_size, c_heads)
        q = q.reshape(b, l, self.heads, new_h * new_w, ws * ws, head_dim)

        # (b, l, h, w, c) -> (b, l, new_h, w_size, w, c) -> (b, l, new_h, w_size, new_w, w_size, c)
        k = k.reshape(b, l, new_h, ws, new_w, ws, qkv_dim)
        # (b, l, new_h, w_size, new_w, w_size, c) -> (b, l, new_h, w_size, new_w, w_size, heads, c_heads)
        k = k.reshape(b, l, new_h, ws, new_w, ws, self.heads, head_dim)
        # (b, l, new_h, w_size, new_w, w_size, heads, c_heads) -> (b, l, heads, new_h, new_w, w_size, w_size, c_heads)
        k = k.permute(0, 1, 6, 2, 4, 3, 5, 7).contiguous()
        # (b, l, heads, new_h, new_w, w_size, w_size, c_heads) -> (b, l, heads, new_h*new_w, w_size*w_size, c_heads)
        k = k.reshape(b, l, self.heads, new_h * new_w, ws * ws, head_dim)
                
        # (b, l, h, w, c) -> (b, l, new_h, w_size, w, c) -> (b, l, new_h, w_size, new_w, w_size, c)
        v = v.reshape(b, l, new_h, ws, new_w, ws, qkv_dim)
        # (b, l, new_h, w_size, new_w, w_size, c) -> (b, l, new_h, w_size, new_w, w_size, heads, c_heads)
        v = v.reshape(b, l, new_h, ws, new_w, ws, self.heads, head_dim)
        # (b, l, new_h, w_size, new_w, w_size, heads, c_heads) -> (b, l, heads, new_h, new_w, w_size, w_size, c_heads)
        v = v.permute(0, 1, 6, 2, 4, 3, 5, 7).contiguous()
        # (b, l, heads, new_h, new_w, w_size, w_size, c_heads) -> (b, l, heads, new_h*new_w, w_size*w_size, c_heads)
        v = v.reshape(b, l, self.heads, new_h * new_w, ws * ws, head_dim)

        # (b, l, heads, new_h*new_w, w_size*w_size, w_size*w_size)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Positional Embedding
        if self.relative_pos_embedding:
            flat_table = self.pos_embedding.view(-1)
            pos_enc = torch.index_select(flat_table, 0, self.rel_idx_1d)
            L_sq = self.window_size ** 2
            dots += pos_enc.view(L_sq, L_sq)
        else:
            dots += self.pos_embedding

        # (b, l, heads, new_h*new_w, w_size*w_size, w_size*w_size)
        attn = dots.softmax(dim=-1)

        # (b, l, heads, new_h*new_w, w_size*w_size, c_head)
        out = torch.matmul(attn, v)

        # (b, l, heads, new_h*new_w, w_size*w_size, c_head) -> (b, l, heads, new_h, new_w, w_size, w_size, c_head)
        out = out.reshape(b, l, self.heads, new_h, new_w, ws, ws, head_dim)
        # (b, l, heads, new_h, new_w, w_size, w_size, c_head) -> (b, l, new_h, w_size, new_w, w_size, heads, c_head)
        out = out.permute(0, 1, 3, 5, 4, 6, 2, 7).contiguous()
        # (b, l, new_h, w_size, new_w, w_size, heads, c_head) -> (b, l, new_h*w_size, new_w*w_size, heads*c_head)
        out = out.reshape(b, l, new_h * ws, new_w * ws, self.heads * head_dim)

        # 7. Final Projection
        out = self.to_out(out)

        return out


class PreNormedPyramidWindowAttention(nn.Module):
    """
    Wrapper for PyramidWindowAttention that adds a prenorm step.
    """
    def __init__(self, prenorm_dim, dim, heads, dim_heads, drop_out, window_size,
                 relative_pos_embedding, fuse_method='naive'):
        super().__init__()

        self.prenorm = nn.LayerNorm(prenorm_dim)
        self.pwin = PyramidWindowAttention(dim, heads, dim_heads, drop_out, window_size,
                                           relative_pos_embedding, fuse_method=fuse_method)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prenorm(x)
        x = self.pwin(x)
        return x


class PyramidWindowAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads, drop_out, window_size,
                 relative_pos_embedding, fuse_method='naive'):
        super().__init__()

        assert isinstance(window_size, list)
        assert isinstance(heads, list)
        assert isinstance(dim_heads, list)
        assert len(dim_heads) == len(heads)
        assert fuse_method in ['naive', 'split_attn']

        self.pwmsa = nn.ModuleList([])

        for (head, dim_head, ws) in zip(heads, dim_heads, window_size):
            self.pwmsa.append(BaseWindowAttention(dim,
                                                 head,
                                                 dim_head,
                                                 drop_out,
                                                 ws,
                                                 relative_pos_embedding))
        
        self.fuse_method = fuse_method
        
        # TensorRT Fix: Store the number of windows as a buffer
        num_windows = len(self.pwmsa)
        self.register_buffer('num_windows_tensor', torch.tensor([num_windows], dtype=torch.float32))

        if fuse_method == 'split_attn':
            self.split_attn = SplitAttn(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        window_outputs = torch.stack([wmsa(x) for wmsa in self.pwmsa], dim=0)

        if self.fuse_method == 'naive':
            return window_outputs.to(torch.float32).mean(dim=0)

        else:
            # self.fuse_method == 'split_attn'
            window_list = torch.unbind(window_outputs, dim=0)
            return self.split_attn(window_list)
        