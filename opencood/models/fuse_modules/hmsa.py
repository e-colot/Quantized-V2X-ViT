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
        B, H, W, L, C = x.shape

        all_q = torch.stack([linear(x) for linear in self.q_linears]) # (NumTypes, B, H, W, L, out_C)
        all_k = torch.stack([linear(x) for linear in self.k_linears])
        all_v = torch.stack([linear(x) for linear in self.v_linears])

        num_types = len(self.q_linears)
        out_q = torch.zeros_like(all_q[0])
        out_k = torch.zeros_like(all_k[0])
        out_v = torch.zeros_like(all_v[0])

        for t_idx in range(num_types):
            # Create a mask where types match the current linear layer index
            # mask shape: (B, 1, 1, L, 1)
            mask = (types == t_idx).view(B, 1, 1, L, 1)
            
            # Add the results for this type to the output
            out_q += all_q[t_idx] * mask
            out_k += all_k[t_idx] * mask
            out_v += all_v[t_idx] * mask

        return out_q, out_k, out_v

    def get_relation_type_index(self, type1, type2):
        return type1 * self.num_types + type2

    def get_hetero_edge_weights(self, x, types):
        w_att_batch = []
        w_msg_batch = []

        for b in range(x.shape[0]):
            w_att_list = []
            w_msg_list = []

            for i in range(x.shape[-2]):
                w_att_i_list = []
                w_msg_i_list = []

                for j in range(x.shape[-2]):
                    e_type = self.get_relation_type_index(types[b, i],
                                                          types[b, j])
                    w_att_i_list.append(self.relation_att[e_type].unsqueeze(0))
                    w_msg_i_list.append(self.relation_msg[e_type].unsqueeze(0))
                w_att_list.append(torch.cat(w_att_i_list, dim=0).unsqueeze(0))
                w_msg_list.append(torch.cat(w_msg_i_list, dim=0).unsqueeze(0))

            w_att_batch.append(torch.cat(w_att_list, dim=0).unsqueeze(0))
            w_msg_batch.append(torch.cat(w_msg_list, dim=0).unsqueeze(0))

        # (B,M,L,L,C_head,C_head)
        w_att = torch.cat(w_att_batch, dim=0).permute(0, 3, 1, 2, 4, 5)
        w_msg = torch.cat(w_msg_batch, dim=0).permute(0, 3, 1, 2, 4, 5)
        return w_att, w_msg

    def to_out(self, x, types):
        out_batch = []
        for b in range(x.shape[0]):
            out_list = []
            for i in range(x.shape[-2]):
                out_list.append(
                    self.a_linears[types[b, i]](x[b, :, :, i, :].unsqueeze(2)))
            out_batch.append(torch.cat(out_list, dim=2).unsqueeze(0))
        out = torch.cat(out_batch, dim=0)
        return out

    def forward(self, x, mask, prior_encoding):
        # x: (B, L, H, W, C) -> (B, H, W, L, C)
        B, L, H, W, C = x.shape
        M = self.heads
        D = C // M

        # Initial permute - use contiguous for TensorRT stability
        x = x.permute(0, 2, 3, 1, 4).contiguous()
        
        # mask: (B, 1, H, W, L, 1) - aligned for broadcasting
        mask = mask.unsqueeze(1)

        # Handle prior_encoding without list comprehension or split
        # Unpack velocities, dts, types directly from the last dim
        pe_slice = prior_encoding[:, :, 0, 0, :]
        # Using slice indexing is more TensorRT-robust than .split()
        types = pe_slice[:, :, 2].to(torch.int32) 
        # (velocities and dts are extracted if needed elsewhere)

        # Heterogeneous projections
        qkv = self.to_qkv(x, types)
        # w_att: (B, M, L, L, D, D), w_msg: (B, M, L, L, D, D)
        w_att, w_msg = self.get_hetero_edge_weights(x, types)

        q_in, k_in, v_in = qkv
        
        # Split heads (Manual rearrange)
        q = q_in.view(B, H, W, L, M, D).permute(0, 4, 1, 2, 3, 5).contiguous()
        k = k_in.view(B, H, W, L, M, D).permute(0, 4, 1, 2, 3, 5).contiguous()
        v = v_in.view(B, H, W, L, M, D).permute(0, 4, 1, 2, 3, 5).contiguous()

        # --- HETEROGENEOUS ATTENTION MAP ---
        # Original: einsum('b m h w i p, b m i j p q, b m h w j q -> b m h w i j', [q, w_att, k])
        # TensorRT Fix: Break into two steps for better optimization and TRT support
        
        # Step 1: Query @ Edge Weight
        # (B, M, H, W, L_i, D_p) * (B, M, L_i, L_j, D_p, D_q) -> (B, M, H, W, L_i, L_j, D_q)
        # Note: We use einsum but avoid the list-input syntax.
        q_w = torch.einsum('bmhwip,bmijpq->bmhwijq', q, w_att)
        
        # Step 2: (Query*Weight) @ Key
        # (B, M, H, W, L_i, L_j, D_q) * (B, M, H, W, L_j, D_q) -> (B, M, H, W, L_i, L_j)
        att_map = torch.einsum('bmhwijq,bmhwjq->bmhwij', q_w, k) * self.scale

        # Masking: Use a large constant instead of float('inf') for FP16 safety
        att_map = att_map.masked_fill(mask == 0, -1e9)
        att_map = self.attend(att_map)

        # --- HETEROGENEOUS MESSAGE PASSING ---
        # Step 1: Edge Weight @ Value
        # (B, M, L_i, L_j, D_p, D_c) * (B, M, H, W, L_j, D_p) -> (B, M, H, W, L_i, L_j, D_c)
        v_msg = torch.einsum('bmijpc,bmhwjp->bmhwijc', w_msg, v)
        
        # Step 2: Attention Map @ Weighted Value
        # (B, M, H, W, L_i, L_j) * (B, M, H, W, L_i, L_j, D_c) -> (B, M, H, W, L_i, D_c)
        out = torch.einsum('bmhwij,bmhwijc->bmhwic', att_map, v_msg)

        # Merge heads (Manual rearrange)
        out = out.permute(0, 2, 3, 4, 1, 5).contiguous()
        out = out.view(B, H, W, L, C)
        
        # Heterogeneous output projection
        out = self.to_out(out, types)
        out = self.drop_out(out)
        
        # Final permute back to (B, L, H, W, C)
        out = out.permute(0, 3, 1, 2, 4).contiguous()
        
        return out