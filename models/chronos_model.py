# models/chronos_model.py
"""
TD-PAU: Temporally-Decayed Path Aggregation Unit

DaeMon zaifligini yechadi:
  DaeMon: K ta snapshot ketma-ket, barchasi teng ahamiyat
           H_1 → gate → H_2 → gate → ... → H_K  (K ta PAU call)

  TD-PAU: barcha snapshot edgelari birlashtiriladi,
          har edge o'z temporal decay weight oladi (1 ta PAU call):
           w(e) = exp(−λ · δt)   δt = (t_query − t_snap) / T_max
           λ — learnable global decay rate

Nima yangi:
  1. Temporal decay on messages (hech bir TKG modelida yo'q)
  2. Single-pass multi-snapshot PAU (DaeMon K call → biz 1 call)
  3. Learnable decay rate (dataset temporal pattern ga moslashadi)
  4. 1-N Cross-Entropy (BCE+64neg dan yaxshi, proven)

Nima DaeMon bilan bir xil (proven, o'zgartirmaslik):
  - query_emb: Embedding(2R, d)
  - PAU: distmult message passing
  - H_init: one-hot × query
  - Score: H[b,e] · query[b]
"""
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SnapGraph = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]


# ─────────────────────────────────────────────────────────────────────────────
# TD-PAU Layer
# ─────────────────────────────────────────────────────────────────────────────

class PAULayer(nn.Module):
    """
    Path Aggregation Unit + Temporal Decay support.

    Standard PAU (DaeMon):
      msg(j→i) = H[j] ⊙ (W_q·q ⊙ w_rel)  +  H_init[j]

    TD-PAU (ours):
      msg(j→i) = w(e) · (H[j] ⊙ (W_q·q ⊙ w_rel)  +  H_init[j])
      w(e) = temporal decay weight for edge e

    Aggregation: weighted mean scatter_add (non-in-place, gradient safe)
    Update:      LN(ReLU(Linear(cat(H, agg))))
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
        H:             torch.Tensor,            # (B, N, d)
        H_init:        torch.Tensor,            # (B, N, d)
        query:         torch.Tensor,            # (B, d)
        src:           torch.Tensor,            # (E,)
        rel:           torch.Tensor,            # (E,)
        dst:           torch.Tensor,            # (E,)
        N:             int,
        edge_weights:  Optional[torch.Tensor] = None,   # (E,) float
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
        w_rel  = self.rel_emb(rel).float()                        # (E, d)
        w_q    = self.query_proj(query).float()                   # (B, d)

        # Distmult message
        src_h  = H[:, src, :].float()                             # (B, E, d)
        init_s = H_init[:, src, :].float()                        # (B, E, d)
        w_comb = w_q.unsqueeze(1) * w_rel.unsqueeze(0)           # (B, E, d)
        msg    = src_h * w_comb + init_s                          # (B, E, d)

        # Temporal decay weighting
        if edge_weights is not None:
            w = edge_weights.float().view(1, E, 1)                # (1, E, 1)
            msg = msg * w                                         # (B, E, d)

        # Non-in-place scatter_add (gradient safe)
        dst_exp = dst.view(1, E, 1).expand(B, E, d)
        agg_sum = torch.zeros(B, N, d, device=device, dtype=torch.float32)
        agg_sum = agg_sum.scatter_add(1, dst_exp, msg)

        # Weighted degree normalization
        with torch.no_grad():
            w_deg = edge_weights.float() if edge_weights is not None \
                    else torch.ones(E, device=device)
            deg = torch.zeros(N, device=device, dtype=torch.float32)
            deg.scatter_add_(0, dst, w_deg)
            deg = deg.clamp(min=1e-6).view(1, N, 1)

        agg = agg_sum / deg
        return self.norm(self.drop(F.relu(
            self.update(torch.cat([H, agg], dim=-1))
        )))


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class CHRONOSModel(nn.Module):
    """
    TD-PAU model: DaeMon + temporal decay weights + 1-N CE.

    Asosiy farq:
      DaeMon: for snap in snapshots: H = PAU(gate(H, H_init), snap)  ← K calls
      TD-PAU: H = PAU(H_init, combined_edges_with_decay)             ← 1 call

    Har edge ning ta'siri: w = exp(−λ · δt)
    λ = learnable (optimal decay rate auto-tuned per dataset)
    """

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

        # Query embedding (DaeMon)
        self.query_emb = nn.Embedding(R, d)
        nn.init.xavier_uniform_(self.query_emb.weight)

        # TD-PAU layers
        self.pau_layers = nn.ModuleList([
            PAULayer(dim=d, num_rels=R, dropout=dropout)
            for _ in range(num_pau_layers)
        ])

        # Learnable temporal decay rate
        # exp(-softplus(λ) · δt)
        # softplus ensures λ > 0
        # init = 1.0 → moderate decay: exp(-1·0.5) ≈ 0.61 at mid-range
        self.decay_rate = nn.Parameter(torch.tensor(1.0))

    # ─────────────────────────────────────────────────────────────────────────

    def _fix_rel(self, rel: torch.Tensor) -> torch.Tensor:
        return rel.clamp(0, self.total_relations - 1)

    def _compute_memory(
        self,
        subjects:        torch.Tensor,       # (B,)
        query:           torch.Tensor,        # (B, d)
        t_query:         float,               # query timestamp (scalar)
        snapshot_graphs: List[SnapGraph],
        device:          torch.device,
    ) -> torch.Tensor:                        # (B, N, d)
        """
        TD-PAU forward pass.

        1. Barcha K snapshot edgelari birlashtiriladi
        2. Har edge: w(e) = exp(−λ · δt)  δt ∈ [0,1]
        3. Bitta PAU call — weighted mean aggregation
        """
        B = subjects.size(0)
        N = self.num_entities
        d = self.entity_dim
        T = float(max(self.num_times, 1))

        # H_init: differentiable one-hot × query
        # H_init[b, subjects[b], :] = query[b, :]
        one_hot = torch.zeros(B, N, device=device, dtype=torch.float32)
        one_hot.scatter_(1, subjects.unsqueeze(1).long(), 1.0)
        H_init = one_hot.unsqueeze(-1) * query.float().unsqueeze(1)  # (B, N, d)

        if not snapshot_graphs:
            return H_init

        # Learnable decay rate (always positive)
        λ = F.softplus(self.decay_rate)                              # scalar > 0

        # Combine all snapshot edges with decay weights
        all_src, all_rel, all_dst, all_w = [], [], [], []

        for snap_src, snap_rel, snap_dst, snap_t in snapshot_graphs:
            snap_src = snap_src.to(device)
            snap_rel = snap_rel.to(device)
            snap_dst = snap_dst.to(device)

            E_k = snap_src.size(0)
            if E_k == 0:
                continue

            # Temporal distance: δt ∈ [0, 1]
            delta = max((t_query - float(snap_t)) / T, 0.0)

            # Decay weight — differentiable through λ
            w_k = torch.exp(-λ * delta)                              # scalar
            w_vec = w_k.expand(E_k)                                  # (E_k,)

            all_src.append(snap_src)
            all_rel.append(snap_rel)
            all_dst.append(snap_dst)
            all_w.append(w_vec)

        if not all_src:
            return H_init

        # Combined edge list
        combined_src = torch.cat(all_src)                            # (E_total,)
        combined_rel = torch.cat(all_rel)
        combined_dst = torch.cat(all_dst)
        combined_w   = torch.cat(all_w)                              # (E_total,)

        # Single-pass TD-PAU
        H = H_init.clone()
        q_f32 = query.float()
        for layer in self.pau_layers:
            H = layer(H, H_init, q_f32,
                      combined_src, combined_rel, combined_dst,
                      N, combined_w)

        return H    # (B, N, d)

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
        t_query   = float(times[0].item())

        memory = self._compute_memory(subjects, query, t_query, snapshot_graphs, device)
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
        t_query   = float(times[0].item())
        memory    = self._compute_memory(subjects, query, t_query, snapshot_graphs, device)
        return self._score_all(memory, query).clamp(-30.0, 30.0)
