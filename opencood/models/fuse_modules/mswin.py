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

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + \
                                    window_size - 1
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
        b, l, h, w, c = x.shape
        m = self.heads
        wh = self.window_size
        ww = self.window_size
        new_h = h // wh
        new_w = w // ww

        # 1. Project and Chunk (Avoids map/lambda)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q_in, k_in, v_in = qkv
        c_head = q_in.shape[-1] // m

        # 2. Window Partition (TRT-safe rearrange alternative)
        # b l (new_h wh) (new_w ww) (m c) -> b l m (new_h new_w) (wh ww) c
        def partition(t):
            # Split dims
            t = t.view(b, l, new_h, wh, new_w, ww, m, c_head)
            # Permute: b(0), l(1), m(6), new_h(2), new_w(4), wh(3), ww(5), c_head(7)
            t = t.permute(0, 1, 6, 2, 4, 3, 5, 7).contiguous()
            # Merge: b, l, m, (new_h*new_w), (wh*ww), c_head
            return t.view(b, l, m, new_h * new_w, wh * ww, c_head)

        q = partition(q_in)
        k = partition(k_in)
        v = partition(v_in)

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
        # b l m (new_h new_w) (wh ww) c -> b l (new_h wh) (new_w ww) (m c)
        
        # Step A: Split back into individual dims
        out = out.view(b, l, m, new_h, new_w, wh, ww, c_head)
        # Step B: Permute to original spatial order
        # Current: 0:b, 1:l, 2:m, 3:new_h, 4:new_w, 5:wh, 6:ww, 7:c_head
        # Target: b(0), l(1), new_h(3), wh(5), new_w(4), ww(6), m(2), c_head(7)
        out = out.permute(0, 1, 3, 5, 4, 6, 2, 7).contiguous()
        # Step C: Collapse into final shape
        out = out.view(b, l, h, w, m * c_head)

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
        
    def forward(self, x):
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

        self.pwmsa = nn.ModuleList([])

        for (head, dim_head, ws) in zip(heads, dim_heads, window_size):
            self.pwmsa.append(BaseWindowAttention(dim,
                                                  head,
                                                  dim_head,
                                                  drop_out,
                                                  ws,
                                                  relative_pos_embedding))
        self.fuse_mehod = fuse_method
        if fuse_method == 'split_attn':
            self.split_attn = SplitAttn(256)

    def forward(self, x):
        output = None
        # naive fusion will just sum up all window attention output and do a
        # mean
        if self.fuse_mehod == 'naive':
            for wmsa in self.pwmsa:
                output = wmsa(x) if output is None else output + wmsa(x)
            return output / len(self.pwmsa)

        elif self.fuse_mehod == 'split_attn':
            window_list = []
            for wmsa in self.pwmsa:
                window_list.append(wmsa(x))
            return self.split_attn(window_list)