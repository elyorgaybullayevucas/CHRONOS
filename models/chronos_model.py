# models/chronos_model.py
"""
DaeMon + 1-N Cross-Entropy

DaeMon arxitekturasi aynan saqlanadi:
  - Entity-free path memory  H ∈ R^{B × N × d}
  - PAU: query-aware distmult message passing (scatter_add, DGL yo'q)
  - MPS: tawaregate  H = σ(W·M) * H_init + (1 - σ(W·M)) * M_prev
  - Score: memory[b,e] · query[b]

Yagona farq DaeMon dan:
  - BCE + 64 neg  →  1-N Cross-Entropy   (barcha entity, to'liq gradient)

Nima uchun 1-N CE yaxshi:
  - DaeMon BCE+64neg: faqat 64 negative entity gradient oladi
  - 1-N CE: barcha N entity gradient oladi → to'liq ma'lumot
  - RE-GCN, HGLS, TiRGN — barchasi 1-N CE bilan DaeMon dan yaxshi
"""
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SnapGraph = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]


# ─────────────────────────────────────────────────────────────────────────────
# PAU Layer  — DaeMon dan aynan
# ─────────────────────────────────────────────────────────────────────────────

class PAULayer(nn.Module):
    """
    Path Aggregation Unit (DaeMon, Dong et al. 2023).

    msg(j→i) = H[j] ⊙ (W_q·query ⊙ w_rel[e])  +  H_init[j]
    agg(i)   = mean scatter_add  (non-in-place — gradient safe)
    H_new(i) = LN(ReLU(Linear(cat(H[i], agg(i)))))
    """

    def __init__(self, dim: int, num_rels: int, dropout: float = 0.1):
        super().__init__()
        self.dim      = dim
        self.num_rels = num_rels

        self.rel_emb    = nn.Embedding(num_rels + 1, dim, padding_idx=num_rels)
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.update     = nn.Linear(dim * 2, dim)
        self.norm       = nn.LayerNorm(dim)
        self.drop       = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.rel_emb.weight[:-1])

    def forward(
        self,
        H:      torch.Tensor,   # (B, N, d)
        H_init: torch.Tensor,   # (B, N, d)
        query:  torch.Tensor,   # (B, d)
        src:    torch.Tensor,   # (E,)
        rel:    torch.Tensor,   # (E,)
        dst:    torch.Tensor,   # (E,)
        N:      int,
    ) -> torch.Tensor:

        B, _, d = H.shape
        E       = src.size(0)
        device  = H.device

        if E == 0:
            agg = torch.zeros(B, N, d, device=device, dtype=torch.float32)
            return self.norm(self.drop(F.relu(
                self.update(torch.cat([H, agg], dim=-1))
            )))

        rel    = rel.clamp(0, self.num_rels - 1)
        w_rel  = self.rel_emb(rel).float()                       # (E, d)
        w_q    = self.query_proj(query).float()                  # (B, d)

        src_h  = H[:, src, :].float()                            # (B, E, d)
        init_s = H_init[:, src, :].float()                       # (B, E, d)
        w_comb = w_q.unsqueeze(1) * w_rel.unsqueeze(0)          # (B, E, d)
        msg    = src_h * w_comb + init_s                         # (B, E, d)

        dst_exp = dst.view(1, E, 1).expand(B, E, d)
        agg_sum = torch.zeros(B, N, d, device=device, dtype=torch.float32)
        agg_sum = agg_sum.scatter_add(1, dst_exp, msg)

        with torch.no_grad():
            deg = torch.zeros(N, device=device, dtype=torch.float32)
            deg.scatter_add_(0, dst, torch.ones(E, device=device))
            deg = deg.clamp(min=1.0).view(1, N, 1)

        agg = agg_sum / deg
        return self.norm(self.drop(F.relu(
            self.update(torch.cat([H, agg], dim=-1))
        )))


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class CHRONOSModel(nn.Module):
    """DaeMon arxitekturasi + 1-N Cross-Entropy."""

    def __init__(
        self,
        num_entities:    int,
        num_relations:   int,
        num_times:       int,
        entity_dim:      int   = 64,
        relation_dim:    int   = 64,
        hidden_dim:      int   = 256,
        delta_dim:       int   = 64,
        dropout:         float = 0.1,
        label_smoothing: float = 0.1,
        use_history:     bool  = True,
        max_history:     int   = 10,
        num_pau_layers:  int   = 2,
        **kwargs,
    ):
        super().__init__()
        self.num_entities       = num_entities
        self.num_base_relations = num_relations
        self.total_relations    = num_relations * 2
        self.num_times          = max(num_times, 1)
        self.entity_dim         = entity_dim
        self.label_smoothing    = label_smoothing
        self.max_history        = max_history

        d = entity_dim
        R = self.total_relations

        # Query embedding
        self.query_emb = nn.Embedding(R, d)
        nn.init.xavier_uniform_(self.query_emb.weight)

        # PAU layers
        self.pau_layers = nn.ModuleList([
            PAULayer(dim=d, num_rels=R, dropout=dropout)
            for _ in range(num_pau_layers)
        ])

        # DaeMon tawaregate: σ(W · memory)
        self.gate_weight = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.gate_weight.weight)
        nn.init.zeros_(self.gate_weight.bias)

    # ─────────────────────────────────────────────────────────────────────────

    def _fix_rel(self, rel: torch.Tensor) -> torch.Tensor:
        return rel.clamp(0, self.total_relations - 1)

    def _compute_memory(
        self,
        subjects:        torch.Tensor,       # (B,)
        query:           torch.Tensor,        # (B, d)
        snapshot_graphs: List[SnapGraph],
        device:          torch.device,
    ) -> torch.Tensor:                        # (B, N, d)
        """
        DaeMon Memory Passing Strategy (tawaregate).

        H_init[b, subjects[b], :] = query[b]   (one-hot × query)
        For each snapshot:
            gate = σ(W · memory)
            H    = gate * H_init + (1-gate) * memory
            H    = PAU(H, H_init, query, snapshot)
            memory = H
        """
        B = subjects.size(0)
        N = self.num_entities
        d = self.entity_dim

        # H_init: differentiable one-hot × query
        one_hot = torch.zeros(B, N, device=device, dtype=torch.float32)
        one_hot.scatter_(1, subjects.unsqueeze(1).long(), 1.0)
        H_init = one_hot.unsqueeze(-1) * query.float().unsqueeze(1)  # (B, N, d)

        if not snapshot_graphs:
            return H_init

        memory   = H_init.clone()
        is_first = True

        for snap_src, snap_rel, snap_dst, _ in snapshot_graphs:
            snap_src = snap_src.to(device)
            snap_rel = snap_rel.to(device)
            snap_dst = snap_dst.to(device)

            if is_first:
                H        = H_init.clone()
                is_first = False
            else:
                # DaeMon tawaregate
                gate = torch.sigmoid(self.gate_weight(memory.float()))
                H    = gate * H_init + (1.0 - gate) * memory

            q_f32 = query.float()
            for layer in self.pau_layers:
                H = layer(H, H_init, q_f32, snap_src, snap_rel, snap_dst, N)

            memory = H

        return memory    # (B, N, d)

    def _score_all(
        self,
        memory: torch.Tensor,   # (B, N, d)
        query:  torch.Tensor,   # (B, d)
    ) -> torch.Tensor:          # (B, N)
        return (memory * query.float().unsqueeze(1)).sum(-1)

    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        subjects:        torch.Tensor,
        relations:       torch.Tensor,
        objects:         torch.Tensor,
        times:           torch.Tensor,
        snapshot_graphs: List[SnapGraph],
        paths=None, path_masks=None, neg_objects=None,
        history=None, hist_mask=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        device    = subjects.device
        rel_fixed = self._fix_rel(relations)
        query     = self.query_emb(rel_fixed).float()

        memory = self._compute_memory(subjects, query, snapshot_graphs, device)
        scores = self._score_all(memory, query)

        link_loss = F.cross_entropy(
            scores, objects,
            label_smoothing=self.label_smoothing,
        )

        losses = {
            "link":     link_loss,
            "ortho":    link_loss.new_tensor(0.0),
            "self_adv": link_loss.new_tensor(0.0),
        }
        return scores, losses

    @torch.no_grad()
    def predict(
        self,
        subjects:        torch.Tensor,
        relations:       torch.Tensor,
        times:           torch.Tensor,
        snapshot_graphs: List[SnapGraph],
        paths=None, path_masks=None, history=None, hist_mask=None,
    ) -> torch.Tensor:

        device    = subjects.device
        rel_fixed = self._fix_rel(relations)
        query     = self.query_emb(rel_fixed).float()
        memory    = self._compute_memory(subjects, query, snapshot_graphs, device)
        return self._score_all(memory, query).clamp(-30.0, 30.0)
