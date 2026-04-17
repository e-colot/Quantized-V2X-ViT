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
        # Flatten and gather weights for each token: (B*L, out_dim, in_dim)
        flat_types = types.view(-1).to(torch.int32)
        w = weight[flat_types] 
        b = bias[flat_types].unsqueeze(-1)
        
        # Reshape x for batch matrix multiplication
        # (B, H, W, L, C) -> (B, L, H, W, C) -> (B*L, H*W, C)
        x_flat = x.permute(0, 3, 1, 2, 4).reshape(x.shape[0] * x.shape[3],
                              x.shape[1] * x.shape[2],
                              x.shape[4])
        
        # out = x @ W.T + b
        # (B*L, H*W, C) @ (B*L, C, out_C) -> (B*L, H*W, out_C)
        out = torch.bmm(x_flat, w.transpose(-1, -2)) + b.transpose(-1, -2)

        # (B*L, H*W, out_C) -> (B, L, H, W, out_C)
        out = out.reshape(x.shape[0], x.shape[3], x.shape[1], x.shape[2], -1)
        # (B, L, H, W, out_C) -> (B, H, W, L, out_C)
        return out.permute(0, 2, 3, 1, 4).contiguous()

    def to_qkv(self, x, types):
        q = self.apply_type_linear(x, types, self.q_weight, self.q_bias)
        k = self.apply_type_linear(x, types, self.k_weight, self.k_bias)
        v = self.apply_type_linear(x, types, self.v_weight, self.v_bias)
        return q, k, v

    def to_out(self, x, types):
        return self.apply_type_linear(x, types, self.a_weight, self.a_bias)

    def get_hetero_edge_weights(self, types):
        t1 = types.unsqueeze(2) 
        t2 = types.unsqueeze(1)
        relation_idx = (t1 * torch.tensor(self.num_types, dtype=torch.int32, device='cuda') + t2).view(-1)
        
        w_att = self.relation_att[relation_idx]
        w_att = w_att.view(types.shape[0], types.shape[1], types.shape[1], self.heads, -1, self.relation_att.shape[-1])
        w_msg = self.relation_msg[relation_idx].view(w_att.shape)
        
        w_att = w_att.permute(0, 3, 1, 2, 4, 5).contiguous()
        w_msg = w_msg.permute(0, 3, 1, 2, 4, 5).contiguous()
        return w_att, w_msg

    def forward(self, x, mask, prior_encoding):
        # x shape: (B, L, H, W, C)

        x = x.permute(0, 2, 3, 1, 4).contiguous() # (B, H, W, L, C)
        mask = mask.unsqueeze(1) # (B, 1, H, W, L)

        types = prior_encoding[:, :, 0, 0, 2].to(torch.int32)

        q_in, k_in, v_in = self.to_qkv(x, types)
        # all the above are (B, H, W, L, out_C)

        w_att, w_msg = self.get_hetero_edge_weights( types)

        head_dim = q_in.shape[-1] // self.heads

        B, H, W, L = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        # (B, H, W, L, out_C) -> (B, H, W, L, heads, D)
        q = q_in.reshape(B, H, W, L, self.heads, head_dim).permute(0, 4, 1, 2, 3, 5).contiguous()
        k = k_in.reshape(B, H, W, L, self.heads, head_dim).permute(0, 4, 1, 2, 3, 5).contiguous()
        v = v_in.reshape(B, H, W, L, self.heads, head_dim).permute(0, 4, 1, 2, 3, 5).contiguous()

        # Attention Map (Einsum is TRT-friendly in recent versions)
        q_w = torch.einsum('bmhwip,bmijpq->bmhwijq', q, w_att)
        att_map = torch.einsum('bmhwijq,bmhwjq->bmhwij', q_w, k) * self.scale
        
        att_map = att_map.masked_fill(mask == 0, -1e9)
        att_map = self.attend(att_map)

        # Message Passing
        v_msg = torch.einsum('bmijpc,bmhwjp->bmhwijc', w_msg, v)
        out = torch.einsum('bmhwij,bmhwijc->bmhwic', att_map, v_msg)

        # Output projection
        out = out.permute(0, 2, 3, 4, 1, 5).contiguous().view(x.shape)
        out = self.to_out(out, types)
        out = self.drop_out(out)
        
        return out.permute(0, 3, 1, 2, 4).contiguous()
    