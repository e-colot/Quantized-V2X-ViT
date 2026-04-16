"""
Multi-scale window transformer
"""
import torch
import torch.nn as nn
import numpy as np

from opencood.models.sub_modules.split_attn import SplitAttn


def get_relative_distances(window_size):
    indices = torch.tensor(np.array(
        [[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class BaseWindowAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, drop_out, window_size,
                 relative_pos_embedding):
        super().__init__()
        inner_dim = dim_head * heads

        self.register_buffer('heads', torch.tensor(heads, dtype=torch.int32, device='cuda'))
        self.register_buffer('scale', torch.tensor(dim_head ** -0.5, dtype=torch.float32, device='cuda'))
        self.register_buffer('window_size', torch.tensor(window_size, dtype=torch.int32, device='cuda'))
        self.relative_pos_embedding = relative_pos_embedding
        self.register_buffer('relative_indices', torch.empty((0, 0, 2), dtype=torch.int32, device='cuda'))

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = (get_relative_distances(window_size) +
                                     window_size - 1).to(dtype=torch.int32, device='cuda')
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1,
                                                          2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2,
                                                          window_size ** 2))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(drop_out)
        )

    def forward(self, x):
        # x shape: (b, l, h, w, c)
        b, l, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        new_h = h // self.window_size
        new_w = w // self.window_size

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q_in, k_in, v_in = qkv
        c_head = q_in.shape[-1] // self.heads
     
        q = q_in.view(b, l, new_h, self.window_size, new_w, self.window_size, self.heads, c_head).permute(0, 1, 6, 2, 4, 3, 5, 7).contiguous()
        q = q.view(b, l, self.heads, new_h * new_w, self.window_size * self.window_size, c_head)

        k = k_in.view(b, l, new_h, self.window_size, new_w, self.window_size, self.heads, c_head).permute(0, 1, 6, 2, 4, 3, 5, 7).contiguous()
        k = k.view(b, l, self.heads, new_h * new_w, self.window_size * self.window_size, c_head)

        v = v_in.view(b, l, new_h, self.window_size, new_w, self.window_size, self.heads, c_head).permute(0, 1, 6, 2, 4, 3, 5, 7).contiguous()
        v = v.view(b, l, self.heads, new_h * new_w, self.window_size * self.window_size, c_head)

        # 3. Attention Calculation
        # TensorRT likes matmul better than einsum for these specific dims
        # q: (..., i, c), k: (..., j, c) -> k.transpose: (..., c, j)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 4. Positional Embedding
        if self.relative_pos_embedding:
            # Note: Ensure relative_indices was cast to int32 in __init__
            dots += self.pos_embedding[self.relative_indices[:, :, 0],
                                    self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        attn = dots.softmax(dim=-1)

        # 5. Combine Value
        # attn: (..., i, j), v: (..., j, c) -> out: (..., i, c)
        out = torch.matmul(attn, v)

        # 6. Window Reversal (TRT-safe rearrange alternative)
        # b l m (new_h new_w) (self.window_size self.window_size) c -> b l (new_h self.window_size) (new_w self.window_size) (m c)
        
        # Step A: Split back into individual dims
        out = out.view(b, l, self.heads, new_h, new_w, self.window_size, self.window_size, c_head)
        # Step B: Permute to original spatial order
        # Current: 0:b, 1:l, 2:m, 3:new_h, 4:new_w, 5:self.window_size, 6:self.window_size, 7:c_head
        # Target: b(0), l(1), new_h(3), self.window_size(5), new_w(4), self.window_size(6), m(2), c_head(7)
        out = out.permute(0, 1, 3, 5, 4, 6, 2, 7).contiguous()
        # Step C: Collapse into final shape
        out = out.view(b, l, h, w, self.heads * c_head)

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
            return torch.mean(window_outputs, dim=0)

        else:
            # self.fuse_method == 'split_attn'
            window_list = torch.unbind(window_outputs, dim=0)
            return self.split_attn(window_list)
        