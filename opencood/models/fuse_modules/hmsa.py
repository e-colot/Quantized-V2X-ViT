import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange


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
        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        for t in range(num_types):
            self.k_linears.append(nn.Linear(dim, inner_dim))
            self.q_linears.append(nn.Linear(dim, inner_dim))
            self.v_linears.append(nn.Linear(dim, inner_dim))
            self.a_linears.append(nn.Linear(inner_dim, dim))

        self.relation_att = nn.Parameter(
            torch.Tensor(num_relations, heads, dim_head, dim_head))
        self.relation_msg = nn.Parameter(
            torch.Tensor(num_relations, heads, dim_head, dim_head))

        torch.nn.init.xavier_uniform_(self.relation_att)
        torch.nn.init.xavier_uniform_(self.relation_msg)

    def to_qkv(self, x, types):
        # x: (B,H,W,L,C)
        # types: (B,L)
        q = self._project_by_type(x, types, self.q_linears)
        k = self._project_by_type(x, types, self.k_linears)
        v = self._project_by_type(x, types, self.v_linears)
        return q, k, v

    def _project_by_type(self, x, types, layers):
        layer_weights = torch.stack([layer.weight for layer in layers], dim=0)
        layer_biases = torch.stack([layer.bias for layer in layers], dim=0)
        projected = torch.einsum('bhwlc,toc->bhwlto', x, layer_weights)
        projected = projected + layer_biases.view(1, 1, 1, 1, -1,
                                                  layer_biases.shape[-1])
        type_mask = F.one_hot(types.to(torch.long),
                              num_classes=self.num_types).to(projected.dtype)
        type_mask = type_mask.unsqueeze(1).unsqueeze(2).unsqueeze(-1)
        return (projected * type_mask).sum(dim=4)

    def get_relation_type_index(self, type1, type2):
        return type1 * self.num_types + type2

    def get_hetero_edge_weights(self, x, types):
        relation_type = self.get_relation_type_index(types[:, :, None],
                                                     types[:, None, :])
        relation_mask = F.one_hot(relation_type.to(torch.long),
                                  num_classes=self.relation_att.shape[0])
        relation_mask = relation_mask.to(x.dtype)
        w_att = torch.einsum('bijr,rhde->bhijde', relation_mask,
                             self.relation_att)
        w_msg = torch.einsum('bijr,rhde->bhijde', relation_mask,
                             self.relation_msg)
        return w_att, w_msg

    def to_out(self, x, types):
        return self._project_by_type(x, types, self.a_linears)

    def forward(self, x, mask, prior_encoding):
        # x: (B, L, H, W, C) -> (B, H, W, L, C)
        # mask: (B, H, W, L, 1)
        # prior_encoding: (B,L,H,W,3)
        x = x.permute(0, 2, 3, 1, 4)
        # mask: (B, 1, H, W, L, 1)
        mask = mask.unsqueeze(1)
        # (B,L)
        velocities, dts, types = [itm.squeeze(-1) for itm in
                                  prior_encoding[:, :, 0, 0, :].split(
                                      [1, 1, 1], dim=-1)]
        types = types.to(torch.int)
        dts = dts.to(torch.int)
        qkv = self.to_qkv(x, types)
        # (B,M,L,L,C_head,C_head)
        w_att, w_msg = self.get_hetero_edge_weights(x, types)

        # q: (B, M, H, W, L, C)
        q, k, v = map(lambda t: rearrange(t, 'b h w l (m c) -> b m h w l c',
                                          m=self.heads), (qkv))
        # attention, (B, M, H, W, L, L)
        att_map = torch.einsum(
            'b m h w i p, b m i j p q, bm h w j q -> b m h w i j',
            [q, w_att, k]) * self.scale
        # add mask
        att_map = att_map.masked_fill(mask == 0, -float('inf'))
        # softmax
        att_map = self.attend(att_map)

        # out:(B, M, H, W, L, C_head)
        v_msg = torch.einsum('b m i j p c, b m h w j p -> b m h w i j c',
                             w_msg, v)
        out = torch.einsum('b m h w i j, b m h w i j c -> b m h w i c',
                           att_map, v_msg)

        out = rearrange(out, 'b m h w l c -> b h w l (m c)',
                        m=self.heads)
        out = self.to_out(out, types)
        out = self.drop_out(out)
        # (B L H W C)
        out = out.permute(0, 3, 1, 2, 4)
        return out