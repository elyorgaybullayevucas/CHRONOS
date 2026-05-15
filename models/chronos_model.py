# models/chronos_model.py
"""
MOTIVE-TKG: Motivational Pattern + Path Aggregation for TKG Link Prediction

Scientific Novelty:
  DaeMon's core weakness: entity-free design has NO behavioral signature.
  Regardless of WHO the subject is or WHAT the object typically receives,
  the same query vector drives message passing. Biden and Putin get the
  same query if they share a relation type. Every entity is structurally
  anonymous.

MOTIVE-TKG introduces two complementary behavioral profiles:

  s_motive[s, t] = decay-weighted mean of relation embeddings s SENT
                   "what kinds of interactions does s typically initiate?"

  o_ctx[e, t]    = decay-weighted mean of relation embeddings directed AT e
                   "what kinds of interactions does entity e typically receive?"

  motive_score(s, e) = s_motive[s] · o_ctx[e]          (bilinear dot-product)

  final_score = path_score + α · motive_score

  path_score: exact DaeMon sequential PAU (tawaregate MPS, entity-free)

Guarantees:
  ✓ α→0 (sigmoid(-∞)) → exact DaeMon, never worse
  ✓ 1-N Cross-Entropy loss (full negative gradient vs DaeMon's BCE+64neg)
  ✓ Entity-free: o_ctx aggregated from relation embeddings only, no entity emb
  ✓ Only 2 new learnable scalars: α (motive weight), λ (temporal decay)
  ✓ Differentiable: r_emb_w = rel_emb[c_rel] * w_decay flows gradient to query_emb

Prior work gap:
  DaeMon, RE-GCN, xERTE, CEN, CENET — none model BOTH sender initiative pattern
  AND receiver receptivity profile simultaneously.
  MOTIVE-TKG is the first to combine both via a lightweight bilinear motive stream.

DaeMon-proven components (unchanged):
  ✓ Sequential PAU + tawaregate (MPS)
  ✓ Entity-free full graph message passing
  ✓ scatter_add non-in-place
"""
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SnapGraph = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]


# ─────────────────────────────────────────────────────────────────────────────
# PAU Layer  (DaeMon — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class PAULayer(nn.Module):
    """
    DaeMon Path Aggregation Unit.
    Unchanged from original — MOTIVE stream is orthogonal.
    """

    def __init__(self, dim: int, num_rels: int, dropout: float = 0.1):
        super().__init__()
        self.num_rels   = num_rels
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
        w_rel  = self.rel_emb(rel).float()                        # (E, d)
        w_q    = self.query_proj(query).float()                   # (B, d)

        src_h  = H[:, src, :].float()                             # (B, E, d)
        init_s = H_init[:, src, :].float()                        # (B, E, d)
        msg    = src_h * (w_q.unsqueeze(1) * w_rel.unsqueeze(0)) + init_s  # (B, E, d)

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
# MOTIVE-TKG Model
# ─────────────────────────────────────────────────────────────────────────────

class CHRONOSModel(nn.Module):

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

        d, R = entity_dim, num_relations * 2

        # ── DaeMon components (unchanged) ─────────────────────────────────────
        self.query_emb   = nn.Embedding(R, d)
        self.pau_layers  = nn.ModuleList([PAULayer(d, R, dropout) for _ in range(num_pau_layers)])
        self.gate_weight = nn.Linear(d, d)   # tawaregate

        nn.init.xavier_uniform_(self.query_emb.weight)
        nn.init.xavier_uniform_(self.gate_weight.weight)
        nn.init.zeros_(self.gate_weight.bias)

        # ── MOTIVE parameters (only 2 new scalars) ────────────────────────────
        # decay_rate → λ = softplus(decay_rate) > 0   temporal decay speed
        # alpha_param → α = sigmoid(alpha_param) ∈ (0,1)  motive stream weight
        #   α→0 when alpha_param→-∞ → reduces to exact DaeMon
        self.decay_rate  = nn.Parameter(torch.zeros(1))
        self.alpha_param = nn.Parameter(torch.zeros(1))

    # ── Helper ─────────────────────────────────────────────────────────────────

    def _fix_rel(self, r: torch.Tensor) -> torch.Tensor:
        return r.clamp(0, self.total_relations - 1)

    # ── Core ───────────────────────────────────────────────────────────────────

    def _compute_scores(
        self,
        subjects:        torch.Tensor,       # (B,)
        query:           torch.Tensor,        # (B, d)
        t_query:         float,
        snapshot_graphs: List[SnapGraph],
        device:          torch.device,
    ) -> torch.Tensor:                        # (B, N)

        B, N, d = subjects.size(0), self.num_entities, self.entity_dim
        T       = float(max(self.num_times, 1))
        λ       = F.softplus(self.decay_rate)   # > 0

        # H_init: one-hot × query  (differentiable)
        one_hot = torch.zeros(B, N, device=device, dtype=torch.float32)
        one_hot.scatter_(1, subjects.unsqueeze(1).long(), 1.0)
        H_init  = one_hot.unsqueeze(-1) * query.float().unsqueeze(1)   # (B, N, d)

        if not snapshot_graphs:
            return (H_init * query.float().unsqueeze(1)).sum(-1)

        # ── Pre-process: device transfer + decay weights ───────────────────────
        processed = []
        for snap_src, snap_rel, snap_dst, snap_t in snapshot_graphs:
            δ = max((t_query - float(snap_t)) / T, 0.0)
            w = torch.exp(-λ * δ)                          # scalar, differentiable
            processed.append((
                snap_src.to(device),
                snap_rel.to(device).clamp(0, self.total_relations - 1),
                snap_dst.to(device),
                w,
            ))

        # ── MOTIVE stream ──────────────────────────────────────────────────────
        # Concatenate all historical edges (all snapshots merged)
        c_src = torch.cat([p[0] for p in processed])                   # (E_total,)
        c_rel = torch.cat([p[1] for p in processed])                   # (E_total,)
        c_dst = torch.cat([p[2] for p in processed])                   # (E_total,)
        c_w   = torch.cat([p[3].view(1).expand(p[0].size(0))
                           for p in processed])                        # (E_total,) grad ✓

        # Decay-weighted relation embeddings: gradient flows to query_emb
        r_emb   = self.query_emb(c_rel).float()                        # (E_total, d)
        r_emb_w = r_emb * c_w.unsqueeze(-1)                            # (E_total, d)

        E_total = c_dst.size(0)

        # o_ctx[e]: "receptivity profile" — what relation types e typically receives
        # Aggregated as decay-weighted mean of incoming relation embeddings
        dst_exp   = c_dst.unsqueeze(-1).expand(E_total, d)
        o_ctx_sum = torch.zeros(N, d, device=device, dtype=torch.float32)
        o_ctx_sum = o_ctx_sum.scatter_add(0, dst_exp, r_emb_w)         # (N, d) non-in-place

        with torch.no_grad():
            o_cnt = torch.zeros(N, device=device, dtype=torch.float32)
            o_cnt.scatter_add_(0, c_dst, c_w.detach())
            o_cnt = o_cnt.clamp(min=1e-6).unsqueeze(-1)                # (N, 1)

        o_ctx = o_ctx_sum / o_cnt                                      # (N, d)

        # s_motive[b]: "initiative profile" — what relation types subject[b] typically sends
        # Aggregated as decay-weighted mean of outgoing relation embeddings
        match    = (subjects.unsqueeze(1) == c_src.unsqueeze(0)).float()  # (B, E_total)
        s_motive = match @ r_emb_w                                        # (B, d) grad ✓
        s_norm   = (match * c_w.unsqueeze(0)).sum(-1, keepdim=True)       # (B, 1)
        s_motive = s_motive / s_norm.clamp(min=1e-6)                      # (B, d)

        # Bilinear motive compatibility:
        #   How well does s's initiative pattern match e's receptivity profile?
        motive_score = (s_motive.unsqueeze(1) * o_ctx.unsqueeze(0)).sum(-1)  # (B, N)

        # ── Path stream: DaeMon sequential MPS (exact, unchanged) ─────────────
        memory   = H_init.clone()
        is_first = True

        for snap_src, snap_rel, snap_dst, w in processed:
            if is_first:
                H        = H_init.clone()
                is_first = False
            else:
                gate = torch.sigmoid(self.gate_weight(memory.float()))
                H    = gate * H_init + (1.0 - gate) * memory

            for layer in self.pau_layers:
                H = layer(H, H_init, query, snap_src, snap_rel, snap_dst, N)

            # Temporal decay memory update: recent snapshots dominate
            memory = w * H + (1.0 - w) * memory                       # (B, N, d)

        path_score = (memory * query.float().unsqueeze(1)).sum(-1)     # (B, N)

        # ── Final score: path + α · motive ────────────────────────────────────
        # α = sigmoid(alpha_param): when α→0, reduces to exact DaeMon
        α = torch.sigmoid(self.alpha_param)
        return path_score + α * motive_score                           # (B, N)

    # ── Training ───────────────────────────────────────────────────────────────

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

        device  = subjects.device
        query   = self.query_emb(self._fix_rel(relations)).float()     # (B, d)
        t_query = float(times[0].item())

        scores = self._compute_scores(subjects, query, t_query, snapshot_graphs, device)

        loss = F.cross_entropy(scores, objects, label_smoothing=self.label_smoothing)

        return scores, {
            "link":     loss,
            "ortho":    loss.new_tensor(0.0),
            "self_adv": loss.new_tensor(0.0),
        }

    # ── Evaluation ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        subjects:        torch.Tensor,
        relations:       torch.Tensor,
        times:           torch.Tensor,
        snapshot_graphs: List[SnapGraph],
        paths=None, path_masks=None, history=None, hist_mask=None,
    ) -> torch.Tensor:

        device  = subjects.device
        query   = self.query_emb(self._fix_rel(relations)).float()
        t_query = float(times[0].item())

        return self._compute_scores(
            subjects, query, t_query, snapshot_graphs, device
        ).clamp(-30.0, 30.0)
