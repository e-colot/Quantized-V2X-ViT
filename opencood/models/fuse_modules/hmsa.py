import torch
from torch import nn

class PreNormedHGTCavAttention(nn.Module):
    """
    Wrapper for HGTCavAttention that adds a prenorm step.
    The dimension of the prenorm is the same as the one given to HGTCavAttention.
    """
    def __init__(self, dim, heads, num_types=2,
                 num_relations=4, dim_head=64, dropout=0.1):
        super().__init__()
        self.prenorm = nn.LayerNorm(dim)
        self.hgtCav = HGTCavAttention(dim, heads, num_types=num_types, num_relations=num_relations,
                                      dim_head=dim_head, dropout=dropout)
        
    def forward(self, x, mask, prior_encoding):
        x = self.prenorm(x)
        x = self.hgtCav(x, mask, prior_encoding)
        return x


class HGTCavAttention(nn.Module):
    def __init__(self, dim, heads, num_types=2,
                 num_relations=4, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.num_types = num_types

        self.attend = nn.Softmax(dim=-1)
        self.drop_out = nn.Dropout(dropout)

        # TRT Optimization: Instead of ModuleList, store weights as a single Parameter
        # (num_types, out_channels, in_channels)
        self.q_weight = nn.Parameter(torch.Tensor(num_types, inner_dim, dim))
        self.q_bias = nn.Parameter(torch.Tensor(num_types, inner_dim))
        self.k_weight = nn.Parameter(torch.Tensor(num_types, inner_dim, dim))
        self.k_bias = nn.Parameter(torch.Tensor(num_types, inner_dim))
        self.v_weight = nn.Parameter(torch.Tensor(num_types, inner_dim, dim))
        self.v_bias = nn.Parameter(torch.Tensor(num_types, inner_dim))
        self.a_weight = nn.Parameter(torch.Tensor(num_types, dim, inner_dim))
        self.a_bias = nn.Parameter(torch.Tensor(num_types, dim))

        self.relation_att = nn.Parameter(torch.Tensor(num_relations, heads, dim_head, dim_head))
        self.relation_msg = nn.Parameter(torch.Tensor(num_relations, heads, dim_head, dim_head))

        self.reset_parameters()

    def reset_parameters(self):
        for w in [self.q_weight, self.k_weight, self.v_weight, self.a_weight, self.relation_att, self.relation_msg]:
            nn.init.xavier_uniform_(w)
        for b in [self.q_bias, self.k_bias, self.v_bias, self.a_bias]:
            nn.init.zeros_(b)

    def apply_type_linear(self, x, types, weight, bias):
        """Vectorized linear projection based on types (B, L)"""
        B, H, W, L, C = x.shape
        # Flatten and gather weights for each token: (B*L, out_dim, in_dim)
        flat_types = types.view(-1)
        w = weight[flat_types] 
        b = bias[flat_types].unsqueeze(-1)
        
        # Reshape x for batch matrix multiplication: (B*L, in_dim, 1)
        # Note: We treat (H, W) as part of the batch for projection
        x_flat = x.permute(0, 3, 1, 2, 4).reshape(B * L, H * W, C)
        
        # out = x @ W.T + b
        # (B*L, HW, C) @ (B*L, C, out_C) -> (B*L, HW, out_C)
        out = torch.bmm(x_flat, w.transpose(-1, -2)) + b.transpose(-1, -2)
        
        # Reshape back to (B, H, W, L, out_C)
        return out.view(B, L, H, W, -1).permute(0, 2, 3, 1, 4).contiguous()

    def to_qkv(self, x, types):
        q = self.apply_type_linear(x, types, self.q_weight, self.q_bias)
        k = self.apply_type_linear(x, types, self.k_weight, self.k_bias)
        v = self.apply_type_linear(x, types, self.v_weight, self.v_bias)
        return q, k, v

    def to_out(self, x, types):
        return self.apply_type_linear(x, types, self.a_weight, self.a_bias)

    def get_hetero_edge_weights(self, x, types):
        B, L = types.shape
        t1 = types.unsqueeze(2) 
        t2 = types.unsqueeze(1)
        relation_idx = (t1 * self.num_types + t2).view(-1)
        
        w_att = self.relation_att[relation_idx].view(B, L, L, self.heads, -1, self.relation_att.shape[-1])
        w_msg = self.relation_msg[relation_idx].view(B, L, L, self.heads, -1, self.relation_msg.shape[-1])
        
        w_att = w_att.permute(0, 3, 1, 2, 4, 5).contiguous()
        w_msg = w_msg.permute(0, 3, 1, 2, 4, 5).contiguous()
        return w_att, w_msg

    def forward(self, x, mask, prior_encoding):
        B, L, H, W, C = x.shape
        M, D = self.heads, C // self.heads

        x = x.permute(0, 2, 3, 1, 4).contiguous() # (B, H, W, L, C)
        mask = mask.unsqueeze(1) # (B, 1, H, W, L)

        # TRT Fix: Use simple indexing to avoid .split() or .item()
        types = prior_encoding[:, :, 0, 0, 2].to(torch.int32)

        q_in, k_in, v_in = self.to_qkv(x, types)
        w_att, w_msg = self.get_hetero_edge_weights(x, types)

        # Reshape for multi-head attention
        q = q_in.view(B, H, W, L, M, D).permute(0, 4, 1, 2, 3, 5).contiguous()
        k = k_in.view(B, H, W, L, M, D).permute(0, 4, 1, 2, 3, 5).contiguous()
        v = v_in.view(B, H, W, L, M, D).permute(0, 4, 1, 2, 3, 5).contiguous()

        # Attention Map (Einsum is TRT-friendly in recent versions)
        q_w = torch.einsum('bmhwip,bmijpq->bmhwijq', q, w_att)
        att_map = torch.einsum('bmhwijq,bmhwjq->bmhwij', q_w, k) * self.scale
        
        att_map = att_map.masked_fill(mask == 0, -1e9)
        att_map = self.attend(att_map)

        # Message Passing
        v_msg = torch.einsum('bmijpc,bmhwjp->bmhwijc', w_msg, v)
        out = torch.einsum('bmhwij,bmhwijc->bmhwic', att_map, v_msg)

        # Output projection
        out = out.permute(0, 2, 3, 4, 1, 5).contiguous().view(B, H, W, L, C)
        out = self.to_out(out, types)
        out = self.drop_out(out)
        
        return out.permute(0, 3, 1, 2, 4).contiguous()
    