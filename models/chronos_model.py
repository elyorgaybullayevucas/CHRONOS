# models/chronos_model.py
"""
QUEST — Query-conditioned Entity-free Snapshot Transformer

DaeMon arxitekturasi ustiga 2 ta ilmiy yangilik:

╔══════════════════════════════════════════════════════════════════╗
║  1. Query-Conditioned Gate (QCG)                 [ASOSIY]       ║
║                                                                  ║
║  DaeMon:  gate = σ(W·M)                                         ║
║           ↑ entity memory ga qaraydi, query bilmaydi            ║
║                                                                  ║
║  QUEST:   gate = σ( W_q(q) ⊙ W_m(M) / √d )                    ║
║           ↑ har entity uchun query-entity o'zaro ta'sir         ║
║                                                                  ║
║  "Attack" query  → tajovuzkor entitylar historysini saqla       ║
║  "Sign-treaty"   → diplomatik entitylarga fokus qil             ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║  2. Relation-Temporal Importance (RTI)          [IKKINCHI]      ║
║                                                                  ║
║  DaeMon:  last K snapshot — hammasi uniform og'irlik            ║
║                                                                  ║
║  QUEST:   w(r, δt) = σ( f_r(query) · (−δt) )                  ║
║           ↑ har relation o'ziga xos temporal horizon o'rganadi  ║
║                                                                  ║
║  "Attack" r.:   yaqin tarix og'ir  → tez decay                  ║
║  "Is-ally-of":  uzoq tarix ham muhim → sekin decay              ║
╚══════════════════════════════════════════════════════════════════╝

Saqlanadi (DaeMon kuchli tomonlari):
  ✓ Entity-free full graph message passing (PAU)
  ✓ Scatter-add non-in-place (DGL/PyG yo'q)
  ✓ 1-N cross-entropy (BCE+64neg dan yaxshi)
  ✓ Timestamp-ordered training
  ✓ FP16 xavfsiz (barcha hisob float32 ichida)

Paper reference:
  - DaeMon (Dong et al., 2023): asosiy backbone
  - Bahdanau (2015): query-attention intuition (QCG uchun)
  - TimeSFormer, Mamba: temporal weighting ilhomi (RTI uchun)
"""
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# SnapGraph tuple: (src_tensor, rel_tensor, dst_tensor, t_snap_int)
SnapGraph = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]


# ─────────────────────────────────────────────────────────────────────────────
# PAU Layer  — DaeMon dan aynan, proven, o'zgartirmaslik
# ─────────────────────────────────────────────────────────────────────────────

class PAULayer(nn.Module):
    """
    Path Aggregation Unit — DGL/PyG'siz PyTorch scatter implementatsiyasi.

    msg(j→i) = H[j] ⊙ (W_q[b] ⊙ w_rel[e])  +  H_init[j]   (distmult + init)
    agg(i)   = mean-pooled scatter_add (non-in-place — gradient safe)
    H_new[i] = LayerNorm(ReLU(Linear(cat(H[i], agg(i)))))

    Barcha tensor float32 — FP16 autocast ostida xavfsiz.
    """

    def __init__(self, dim: int, num_rels: int, dropout: float = 0.1):
        super().__init__()
        self.dim      = dim
        self.num_rels = num_rels

        # Relation embedding (padding_idx = num_rels)
        self.rel_emb    = nn.Embedding(num_rels + 1, dim, padding_idx=num_rels)
        nn.init.xavier_uniform_(self.rel_emb.weight[:-1])

        # Query projection — query-aware message scaling
        self.query_proj = nn.Linear(dim, dim, bias=False)

        # Update MLP
        self.update = nn.Linear(dim * 2, dim)
        self.norm   = nn.LayerNorm(dim)
        self.drop   = nn.Dropout(dropout)

    def forward(
        self,
        H:      torch.Tensor,   # (B, N, d) float32
        H_init: torch.Tensor,   # (B, N, d) float32  — anchor
        query:  torch.Tensor,   # (B, d)    float32
        src:    torch.Tensor,   # (E,)      long
        rel:    torch.Tensor,   # (E,)      long
        dst:    torch.Tensor,   # (E,)      long
        N:      int,
    ) -> torch.Tensor:          # (B, N, d) float32

        B, _, d = H.shape
        E       = src.size(0)
        device  = H.device

        # Empty snapshot → zero aggregation
        if E == 0:
            agg = torch.zeros(B, N, d, device=device, dtype=torch.float32)
            return self.norm(self.drop(F.relu(self.update(
                torch.cat([H, agg], dim=-1)
            ))))

        # Clamp relation indices
        rel = rel.clamp(0, self.num_rels - 1)

        # Relation weights + query projection
        w_rel = self.rel_emb(rel).float()                        # (E, d)
        w_q   = self.query_proj(query).float()                   # (B, d)

        # Distmult message: H[src] ⊙ (query ⊙ w_rel) + H_init[src]
        src_h  = H[:, src, :].float()                            # (B, E, d)
        init_s = H_init[:, src, :].float()                       # (B, E, d)
        w_comb = w_q.unsqueeze(1) * w_rel.unsqueeze(0)          # (B, E, d)
        msg    = src_h * w_comb + init_s                         # (B, E, d)

        # Non-in-place scatter_add (gradient flows through msg)
        dst_exp = dst.view(1, E, 1).expand(B, E, d)
        agg_sum = torch.zeros(B, N, d, device=device, dtype=torch.float32)
        agg_sum = agg_sum.scatter_add(1, dst_exp, msg)           # (B, N, d)

        # Degree normalization (no gradient needed)
        with torch.no_grad():
            deg = torch.zeros(N, device=device, dtype=torch.float32)
            deg.scatter_add_(0, dst, torch.ones(E, device=device))
            deg = deg.clamp(min=1.0).view(1, N, 1)
        agg = agg_sum / deg                                      # (B, N, d)

        # Update: LN(ReLU(Linear(cat(H, agg))))
        return self.norm(self.drop(F.relu(
            self.update(torch.cat([H, agg], dim=-1))
        )))


# ─────────────────────────────────────────────────────────────────────────────
# QUEST Model
# ─────────────────────────────────────────────────────────────────────────────

class CHRONOSModel(nn.Module):
    """
    QUEST: Entity-free TKG extrapolation with QCG + RTI.

    score(s, r, o, t) = memory[b, o] · query[b]
    memory built via: QCG gate → PAU → RTI-weighted update
    """

    def __init__(
        self,
        num_entities:    int,
        num_relations:   int,           # base relations (inversesiz)
        num_times:       int,
        entity_dim:      int   = 64,
        relation_dim:    int   = 64,    # compat alias, entity_dim ishlatiladi
        hidden_dim:      int   = 256,   # compat, unused
        delta_dim:       int   = 64,    # compat, unused
        dropout:         float = 0.1,
        label_smoothing: float = 0.1,
        use_history:     bool  = True,  # compat, unused
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

        # ── Query embedding (DaeMon style) ───────────────────────────────────
        # Har bir (forward + inverse) relation uchun alohida query vector
        self.query_emb = nn.Embedding(R, d)
        nn.init.xavier_uniform_(self.query_emb.weight)

        # ── PAU layers (DaeMon, proven) ───────────────────────────────────────
        self.pau_layers = nn.ModuleList([
            PAULayer(dim=d, num_rels=R, dropout=dropout)
            for _ in range(num_pau_layers)
        ])

        # ── QCG: Query-Conditioned Gate ───────────────────────────────────────
        # gate[b,e] = σ( W_q(query[b]) ⊙ W_m(memory[b,e]) / √d )
        # query: (B, d) → W_q → (B, d)    [query projection]
        # memory: (B, N, d) → W_m → (B, N, d)  [memory projection]
        # dot-product per entity, sigmoid → (B, N, d) gate
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_m = nn.Linear(d, d, bias=False)
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_m.weight)

        # ── RTI: Relation-Temporal Importance ────────────────────────────────
        # w(r, δt) = σ( decay_r · (−δt) + bias )
        # decay_r = rti_decay(query[b])   ← relation-specific decay rate
        # δt = (t_query - t_snap) / T_max ∈ [0, 1]
        # decay_r > 0 → recent snapshots more important (most relations)
        # decay_r < 0 → distant snapshots more important (rare, e.g. long cycles)
        self.rti_decay = nn.Linear(d, 1)          # (d → 1) decay rate projection
        self.rti_bias  = nn.Parameter(torch.zeros(1))  # global bias
        nn.init.zeros_(self.rti_decay.weight)
        nn.init.zeros_(self.rti_decay.bias)
        # init: decay=0, bias=0 → w=sigmoid(0)=0.5 uniform → safe starting point

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fix_rel(self, rel: torch.Tensor) -> torch.Tensor:
        return rel.clamp(0, self.total_relations - 1)

    # ── Core: Memory computation with QCG + RTI ───────────────────────────────

    def _compute_memory(
        self,
        subjects:        torch.Tensor,        # (B,) long
        query:           torch.Tensor,         # (B, d) float32
        times:           torch.Tensor,         # (B,) long — all same (timestamp-ordered)
        snapshot_graphs: List[SnapGraph],      # [(src,rel,dst,t_snap), ...]
        device:          torch.device,
    ) -> torch.Tensor:                         # (B, N, d) float32
        """
        DaeMon path memory + QCG gate + RTI weighting.

        H_init: one-hot × query (differentiable — gradient flows to query_emb)
        QCG:    query-conditioned blend of H_init and past memory
        RTI:    relation-specific temporal importance for each snapshot
        """
        B = subjects.size(0)
        N = self.num_entities
        d = self.entity_dim

        # ── H_init: differentiable one-hot × query ────────────────────────────
        # one_hot: (B, N) binary, no grad
        # H_init[b, subjects[b], :] = query[b, :]   — others = 0
        # Gradient flows: ∂L/∂query via H_init → H_init has grad ✓
        one_hot = torch.zeros(B, N, device=device, dtype=torch.float32)
        one_hot.scatter_(1, subjects.unsqueeze(1).long(), 1.0)
        H_init = one_hot.unsqueeze(-1) * query.float().unsqueeze(1)  # (B, N, d)

        # ── No history → return H_init ────────────────────────────────────────
        if not snapshot_graphs:
            return H_init

        memory   = H_init.clone()
        t_query  = float(times[0].item())
        T_max    = float(max(self.num_times, 1))

        # ── Process snapshots: oldest → newest ───────────────────────────────
        for snap_src, snap_rel, snap_dst, snap_t in snapshot_graphs:
            snap_src = snap_src.to(device)
            snap_rel = snap_rel.to(device)
            snap_dst = snap_dst.to(device)

            # ── RTI: temporal importance weight ──────────────────────────────
            # δt ∈ [0, 1]: 0 = very recent, 1 = very distant
            delta    = (t_query - float(snap_t)) / T_max
            delta    = max(delta, 0.0)                           # safety clamp

            # decay_r = rti_decay(query) ∈ R  → relation-specific rate
            # w[b] = σ( decay_r[b] · (-δt) + bias )
            # shape: (B, 1)
            decay_r  = self.rti_decay(query.float())             # (B, 1)
            w        = torch.sigmoid(
                decay_r * (-delta) + self.rti_bias
            )                                                    # (B, 1)

            # ── QCG: Query-Conditioned Gate ───────────────────────────────────
            # gate[b,e] = σ( W_q(q[b]) ⊙ W_m(M[b,e]) / √d )
            # → entity-specific, query-aware blend of H_init and memory
            q_proj  = self.W_q(query.float()).unsqueeze(1)       # (B, 1, d)
            m_proj  = self.W_m(memory.float())                   # (B, N, d)
            gate    = torch.sigmoid(
                q_proj * m_proj / math.sqrt(d)
            )                                                    # (B, N, d)
            H = gate * H_init + (1.0 - gate) * memory           # (B, N, d)

            # ── PAU: full graph message passing ───────────────────────────────
            q_f32 = query.float()
            for layer in self.pau_layers:
                H = layer(H, H_init, q_f32, snap_src, snap_rel, snap_dst, N)

            # ── RTI-weighted memory update ─────────────────────────────────────
            # High w (important snapshot) → memory ≈ H  (update heavily)
            # Low w (unimportant)         → memory ≈ old memory (skip)
            w_exp  = w.unsqueeze(-1)                             # (B, 1, 1)
            memory = w_exp * H + (1.0 - w_exp) * memory         # (B, N, d)

        return memory    # (B, N, d) float32

    # ── Scoring (entity-free) ─────────────────────────────────────────────────

    def _score_all(
        self,
        memory: torch.Tensor,   # (B, N, d) float32
        query:  torch.Tensor,   # (B, d)    float32
    ) -> torch.Tensor:          # (B, N)    float32
        """
        score[b, e] = memory[b, e] · query[b]
        Entity-free: no entity lookup, scoring via memory content only.
        """
        return (memory * query.float().unsqueeze(1)).sum(-1)     # (B, N)

    # ── Forward (training) ────────────────────────────────────────────────────

    def forward(
        self,
        subjects:        torch.Tensor,         # (B,) long
        relations:       torch.Tensor,         # (B,) long
        objects:         torch.Tensor,         # (B,) long
        times:           torch.Tensor,         # (B,) long
        snapshot_graphs: List[SnapGraph],
        # Legacy compat kwargs (ignored):
        paths=None, path_masks=None, neg_objects=None,
        history=None, hist_mask=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        device    = subjects.device
        rel_fixed = self._fix_rel(relations)
        query     = self.query_emb(rel_fixed).float()            # (B, d)

        memory = self._compute_memory(subjects, query, times, snapshot_graphs, device)
        scores = self._score_all(memory, query)                  # (B, N)

        # 1-N cross-entropy with label smoothing
        link_loss = F.cross_entropy(
            scores, objects,
            label_smoothing=self.label_smoothing,
        )

        # Orthogonal regularizer — diverse relation representations
        # ‖Q^T Q − I‖_F → encourages orthogonal query vectors
        Q     = self.query_emb.weight.float()
        ortho = torch.norm(
            Q @ Q.T - torch.eye(Q.size(0), device=device), p="fro"
        )

        losses = {
            "link":     link_loss,
            "ortho":    ortho,
            "self_adv": link_loss.new_tensor(0.0),   # compat
        }
        return scores, losses

    # ── Predict (evaluation) ─────────────────────────────────────────────────

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
        memory    = self._compute_memory(subjects, query, times, snapshot_graphs, device)
        return self._score_all(memory, query).clamp(-30.0, 30.0)
