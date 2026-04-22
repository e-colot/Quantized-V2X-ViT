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
        self.register_buffer('scale', torch.tensor(dim_head ** -0.5, dtype=torch.float32, device='cuda'))
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
        r"""
        x: (H, W, L, C)
        types: (L)
        """
        # Flatten and gather weights for each token: (B*L, out_dim, in_dim)
        flat_types = types.to(torch.int32).clamp(min=0)

        w = weight[flat_types] 
        b = bias[flat_types].unsqueeze(-1)
        
        # Reshape x for batch matrix multiplication
        # (H, W, L, C) -> (L, H, W, C) -> (L, H*W, C)
        HW = int(x.shape[0]) * int(x.shape[1])
        L = int(x.shape[2])
        C = int(x.shape[3])
        x_flat = x.permute(2, 0, 1, 3).reshape(L, HW, C)
        
        # out = x @ W.T + b
        # (L, H*W, C) @ (L, C, out_C) -> (L, H*W, out_C)
        out = torch.bmm(x_flat, w.transpose(-1, -2)) + b.transpose(-1, -2)

        # (L, H*W, out_C) -> (L, H, W, out_C)
        out = out.reshape(L, int(x.shape[0]), int(x.shape[1]), -1)
        # (L, H, W, out_C) -> (H, W, L, out_C)
        return out.permute(1, 2, 0, 3).contiguous()

    def to_qkv(self, x, types):
        r"""
        x: (H, W, L, C)
        types: (L)
        """
        q = self.apply_type_linear(x, types, self.q_weight, self.q_bias)
        k = self.apply_type_linear(x, types, self.k_weight, self.k_bias)
        v = self.apply_type_linear(x, types, self.v_weight, self.v_bias)
        return q, k, v

    def to_out(self, x, types):
        return self.apply_type_linear(x, types, self.a_weight, self.a_bias)

    def get_hetero_edge_weights(self, types):
        r"""
        types: (L)
        """
        # t1: (L, 1)
        t1 = types.unsqueeze(1) 
        # t1: (1, L)
        t2 = types.unsqueeze(0)

        relation_idx = (t1 * torch.tensor(self.num_types, dtype=torch.int32, device='cuda') + t2).view(-1)
        
        w_att = self.relation_att[relation_idx]
        # w_att & w_msg: (L, L, heads, -1, rel_dim)
        w_att = w_att.view(types.shape[0], types.shape[0], self.heads, -1, self.relation_att.shape[-1])
        w_msg = self.relation_msg[relation_idx].view(w_att.shape)
        
        # both (heads, L, L, -1, rel_dim)
        w_att = w_att.permute(2, 0, 1, 3, 4).contiguous()
        w_msg = w_msg.permute(2, 0, 1, 3, 4).contiguous()
        return w_att, w_msg

    def forward(self, x, mask, prior_encoding):
        r"""
        x: (L, H, W, C)
        mask: (H, W, 1, L)
        prior_encoding: (L, H, W, 3)
        """

        x = x.permute(1, 2, 0, 3).contiguous()
        # x: (H, W, L, C)

        types = torch.select(torch.select(torch.select(prior_encoding, 3, 2), 2, 0), 1, 0).to(torch.int32)
        # types: (L)

        q_in, k_in, v_in = self.to_qkv(x, types)
        # all the above are (H, W, L, out_C)

        head_dim = int(q_in.shape[-1]) // self.heads

        H, W, L = x.shape[0], x.shape[1], x.shape[2]

        q = q_in.reshape(H, W, L, self.heads, head_dim).permute(3, 0, 1, 2, 4).contiguous()
        k = k_in.reshape(H, W, L, self.heads, head_dim).permute(3, 0, 1, 2, 4).contiguous()
        v = v_in.reshape(H, W, L, self.heads, head_dim).permute(3, 0, 1, 2, 4).contiguous()
        # (H, W, L, out_C) -> (H, W, L, heads, D) -> (heads, H, W, L, D)

        w_att, w_msg = self.get_hetero_edge_weights(types)
        # both (heads, L, L, -1, rel_dim)

        q_w = torch.einsum('mhwip,mijpq->mhwijq', q, w_att)
        # q_w: (heads, H, W, L, L, rel_dim)
        att_map = torch.einsum('mhwijq,mhwjq->mhwij', q_w, k) * self.scale
        # att_map: (heads, H, W, L, L)
        
        mask = mask.unsqueeze(0)
        # mask: (1, H, W, 1, L)
        att_map = att_map.masked_fill(mask == 0, float(-1e9))
        att_map = self.attend(att_map)
        # att_map: (heads, H, W, L, L)

        # Message Passing
        v_msg = torch.einsum('mijpc,mhwjp->mhwijc', w_msg, v)
        # v_msg: (heads, H, W, L, L, rel_dim)
        out = torch.einsum('mhwij,mhwijc->mhwic', att_map, v_msg)
        # out: (heads, H, W, L, rel_dim)

        # Output projection
        out = out.permute(1, 2, 3, 0, 4).contiguous().view(x.shape)
        # out: (H, W, L, heads*rel_dim = C)
        out = self.to_out(out, types)
        out = self.drop_out(out)
        
        return out.permute(2, 0, 1, 3).contiguous()
        # out: (L, H, W, C)
    