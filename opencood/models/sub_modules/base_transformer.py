import torch
from torch import nn


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        raise NotImplemented('discarded wrapper for tensorrt deployment. Implement a function-specific wrapper to avoid **kwargs')
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNormedFeedForward(nn.Module):
    """
    Wrapper for FeedForward that adds a prenorm step.
    The dimension of the prenorm is the same as the one given to FeedForward.
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.prenorm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, hidden_dim, dropout=dropout)

    def forward(self, x):
        x = self.prenorm(x)
        x = self.ff(x)
        return x
    

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class PreNormedCavAttention(nn.Module):
    """
    Wrapper for Vanilla CAV attention that adds a prenorm step.
    The dimension of the prenorm is the same as the one given to CavAttention.
    """
    def __init__(self, dim, heads, dim_head=64, dropout=0.1):
        super().__init__()
        self.prenorm = nn.LayerNorm(dim)
        self.att = CavAttention(dim, heads, dim_head=dim_head, dropout=dropout)

    def forward(self, x, mask):
        x = self.prenorm(x)
        x = self.att(x, mask)
        return x


class CavAttention(nn.Module):
    """
    Vanilla CAV attention.
    """
    def __init__(self, dim, heads, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask, prior_encoding):
        # x: (B, L, H, W, C) -> (B, H, W, L, C)
        B, L, H, W, C = x.shape
        M = self.heads
        D = C // M  # Head dimension
        
        # Initial permute
        x = x.permute(0, 2, 3, 1, 4).contiguous()
        
        # mask adjustment: (B, L) -> (B, 1, 1, 1, 1, L) 
        # Needs to align with att_map: (B, M, H, W, L, L)
        # We broadcast across the query dimension (i) to mask the keys (j)
        mask = mask.view(B, 1, 1, 1, 1, L)

        # qkv: (B, H, W, L, 3*C) -> 3 tensors of (B, H, W, L, C)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q_in, k_in, v_in = qkv

        # 1. Replace map(lambda rearrange...): Split heads
        # b h w l (m c) -> b m h w l c
        def split_heads(t):
            t = t.view(B, H, W, L, M, D)
            return t.permute(0, 4, 1, 2, 3, 5).contiguous()

        q = split_heads(q_in)
        k = split_heads(k_in)
        v = split_heads(v_in)

        # 2. Replace einsum with matmul for Attention Score
        # (B, M, H, W, L, D) @ (B, M, H, W, D, L) -> (B, M, H, W, L, L)
        att_map = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 3. Add mask (TensorRT handles masked_fill well)
        att_map = att_map.masked_fill(mask == 0, -1e9) # Use large neg number instead of inf
        
        # softmax (attend)
        att_map = self.attend(att_map)

        # 4. Replace einsum with matmul for Value aggregation
        # (B, M, H, W, L, L) @ (B, M, H, W, L, D) -> (B, M, H, W, L, D)
        out = torch.matmul(att_map, v)

        # 5. Replace rearrange: Merge heads
        # b m h w l c -> b h w l (m c)
        out = out.permute(0, 2, 3, 4, 1, 5).contiguous()
        out = out.view(B, H, W, L, C)

        # 6. Final projections and permute back
        out = self.to_out(out) # (B, H, W, L, C)
        
        # (B, H, W, L, C) -> (B, L, H, W, C)
        # Your original code used .permute(0, 3, 1, 2, 4)
        # Let's verify: 0:B, 1:H, 2:W, 3:L, 4:C
        # Result: B, L, H, W, C
        out = out.permute(0, 3, 1, 2, 4).contiguous()
        
        return out


class BaseEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormedCavAttention(dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout),
                PreNormedFeedForward(dim, mlp_dim, dropout=dropout)]))
        # for _ in range(depth):
        #     self.layers.append(nn.ModuleList([
        #         PreNormedCavAttention(dim,
        #                     heads=heads,
        #                     dim_head=dim_head,
        #                     dropout=dropout),
        #         PreNormedFeedForward(dim, mlp_dim, dropout=dropout)]))

    def forward(self, x, mask):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return x


class BaseTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        dim = args['dim']
        depth = args['depth']
        heads = args['heads']
        dim_head = args['dim_head']
        mlp_dim = args['mlp_dim']
        dropout = args['dropout']
        max_cav = args['max_cav']

        self.encoder = BaseEncoder(dim, depth, heads, dim_head, mlp_dim,
                                   dropout)

    def forward(self, x, mask):
        # B, L, H, W, C
        output = self.encoder(x, mask)
        # B, H, W, C
        output = output[:, 0]

        return output