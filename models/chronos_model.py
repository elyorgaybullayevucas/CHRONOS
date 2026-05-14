# models/chronos_model.py
"""
NEXUS — Neural EXtrapolation with Unified Snapshot-graphs

DaeMon dan yaxshiroq model, 3 ta aniq ilmiy yangilik:
  1. DE entity embeddings  — DaeMon entity-free; biz temporal entity
                             modulatsiyasi qo'shamiz (periodic + trend).
                             → Entity-specific temporal patterns
  2. 1-N cross-entropy     — DaeMon BCE+64 neg (sampling ceiling).
                             1-N: barcha E entity ustidan CE → to'liq gradient.
                             → Ma'lumot bottleneck yo'q
  3. Bilinear joint score  — score = (memory + de_emb) · query
                             → Ikki signal kombinatsiyasi

Asosiy arxitektura (DaeMon bilan bir xil):
  - Query embedding: Embedding(2R, d)
  - Path memory: H ∈ R^{B × N × d}
  - PAU: 2-layer query-aware distmult message passing (PyTorch scatter, DGL yoq)
  - MPS: time-aware gate  H = σ(W·M)·H_init + (1-σ(W·M))·M_prev
  - Score: (H[b,e] + de(e,t)) · query[b]
  - Loss: 1-N CE + ortho regularizer

FP16 xavfsizligi:
  - _compute_memory: barcha hisob float32
  - scatter_add: non-in-place (gradient uchun)
  - de_emb: float32 qaytaradi
"""
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal encoding
# ─────────────────────────────────────────────────────────────────────────────

def sinusoidal_enc(t: torch.Tensor, dim: int) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal_enc: dim must be even, got {dim}")
    shape  = t.shape
    t_flat = t.float().reshape(-1, 1).clamp(0, 1e5)
    half   = dim // 2
    div    = torch.exp(
        torch.arange(half, device=t_flat.device, dtype=torch.float32)
        * -(math.log(10000.0) / half)
    )
    args = t_flat * div
    enc  = torch.empty(t_flat.size(0), dim, device=t_flat.device, dtype=torch.float32)
    enc[:, 0::2] = torch.sin(args)
    enc[:, 1::2] = torch.cos(args)
    return enc.reshape(*shape, dim)


# ─────────────────────────────────────────────────────────────────────────────
# DE Entity Embedding
# ─────────────────────────────────────────────────────────────────────────────

class DEEntityEmbedding(nn.Module):
    """
    Diachronic Embedding: periodic + trend temporal modulation.
    Always returns float32.
    """

    def __init__(self, num_entities: int, dim: int):
        super().__init__()
        self.mod_dim = dim // 2
        self.emb   = nn.Embedding(num_entities, dim)
        self.amp   = nn.Embedding(num_entities, self.mod_dim)
        self.freq  = nn.Embedding(num_entities, self.mod_dim)
        self.phase = nn.Embedding(num_entities, self.mod_dim)
        self.trend = nn.Embedding(num_entities, self.mod_dim)
        nn.init.xavier_uniform_(self.emb.weight)
        nn.init.constant_(self.amp.weight,  0.1)
        nn.init.uniform_(self.freq.weight,  0.01, 2.0)
        nn.init.uniform_(self.phase.weight, 0.0, math.pi)
        nn.init.zeros_(self.trend.weight)

    def forward(self, entities: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        """entities: (N,)  t_norm: (N,) → (N, dim) float32"""
        e    = self.emb(entities).float()
        t    = t_norm.float().unsqueeze(-1)
        amp  = torch.sigmoid(self.amp(entities).float())
        frq  = F.softplus(self.freq(entities).float())
        phi  = self.phase(entities).float()
        trd  = self.trend(entities).float()
        mod  = amp * torch.sin(frq * t + phi)
        periodic = e[:, :self.mod_dim] * (1.0 + mod)
        trended  = e[:, self.mod_dim:] + trd * t
        return torch.cat([periodic, trended], dim=-1)   # (N, d) float32


# ─────────────────────────────────────────────────────────────────────────────
# PAU Layer  (scatter_add non-in-place — gradient safe)
# ─────────────────────────────────────────────────────────────────────────────

class PAULayer(nn.Module):
    """
    Path Aggregation Unit — DGL/PyG'siz scatter implementatsiyasi.

    msg(j→i) = H[j] * (W_q[b] ⊙ w_rel[e])  +  H_init[j]
    agg(i)   = mean scatter_add (non-in-place)
    H_new[i] = LN(ReLU(Linear(cat(H[i], agg(i)))))

    Barcha operatsiyalar float32 — FP16 autocast xavfsizligi.
    """

    def __init__(self, dim: int, num_rels: int, dropout: float = 0.1):
        super().__init__()
        self.dim      = dim
        self.num_rels = num_rels
        # Relation embedding
        self.rel_emb    = nn.Embedding(num_rels + 1, dim, padding_idx=num_rels)
        nn.init.xavier_uniform_(self.rel_emb.weight[:-1])
        # Query projection (query-aware)
        self.query_proj = nn.Linear(dim, dim, bias=False)
        # Update
        self.update     = nn.Linear(dim * 2, dim)
        self.norm       = nn.LayerNorm(dim)
        self.drop       = nn.Dropout(dropout)

    def forward(
        self,
        H:       torch.Tensor,   # (B, N, d) float32
        H_init:  torch.Tensor,   # (B, N, d) float32
        query:   torch.Tensor,   # (B, d)    float32
        src:     torch.Tensor,   # (E,) long
        rel:     torch.Tensor,   # (E,) long
        dst:     torch.Tensor,   # (E,) long
        N:       int,
    ) -> torch.Tensor:           # (B, N, d) float32
        B, _, d = H.shape
        E       = src.size(0)
        device  = H.device

        # Empty snapshot — return update with zero agg
        if E == 0:
            agg = torch.zeros(B, N, d, device=device, dtype=torch.float32)
            out = self.update(torch.cat([H, agg], dim=-1))
            return self.norm(self.drop(F.relu(out)))

        # Safety clamp relation indices
        rel = rel.clamp(0, self.num_rels - 1)

        # Relation and query weights
        w_rel = self.rel_emb(rel).float()                              # (E, d)
        w_q   = self.query_proj(query).float()                         # (B, d)

        # Distmult message + H_init augmentation (DaeMon style)
        src_h  = H[:, src, :].float()                                  # (B, E, d)
        init_s = H_init[:, src, :].float()                             # (B, E, d)
        w_comb = w_q.unsqueeze(1) * w_rel.unsqueeze(0)                # (B, E, d)
        msg    = src_h * w_comb + init_s                               # (B, E, d)

        # Non-in-place scatter_add (gradient flows correctly through msg)
        dst_exp = dst.view(1, E, 1).expand(B, E, d)                   # (B, E, d)
        agg_sum = torch.zeros(B, N, d, device=device, dtype=torch.float32)
        agg_sum = agg_sum.scatter_add(1, dst_exp, msg)                 # non-in-place

        # Degree normalization (no grad needed)
        with torch.no_grad():
            count = torch.zeros(N, device=device, dtype=torch.float32)
            count.scatter_add_(0, dst, torch.ones(E, device=device))
            count = count.clamp(min=1.0).view(1, N, 1)
        agg = agg_sum / count                                          # (B, N, d)

        # Update: LN(ReLU(Linear(cat(H, agg))))
        out = self.update(torch.cat([H, agg], dim=-1))                 # (B, N, d)
        return self.norm(self.drop(F.relu(out)))


# ─────────────────────────────────────────────────────────────────────────────
# NEXUS Model
# ─────────────────────────────────────────────────────────────────────────────

SnapGraph = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class CHRONOSModel(nn.Module):
    """
    NEXUS: DaeMon path memory + DE entity embeddings + 1-N CE.
    score(s, r, o, t) = (memory[b,o] + de(o,t)) · query(r)
    """

    def __init__(
        self,
        num_entities:    int,
        num_relations:   int,           # base (inversesiz)
        num_times:       int,
        entity_dim:      int   = 128,
        relation_dim:    int   = 128,   # compat, unused
        hidden_dim:      int   = 256,   # compat, unused
        delta_dim:       int   = 64,    # compat, unused
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

        # ── Query embedding (DaeMon style) ───────────────────────────────────
        self.query_emb = nn.Embedding(R, d)
        nn.init.xavier_uniform_(self.query_emb.weight)

        # ── PAU layers ───────────────────────────────────────────────────────
        self.pau_layers = nn.ModuleList([
            PAULayer(dim=d, num_rels=R, dropout=dropout)
            for _ in range(num_pau_layers)
        ])

        # ── Memory gate (DaeMon tawaregate) ──────────────────────────────────
        self.gate_weight = nn.Linear(d, d)

        # ── DE entity embedding ───────────────────────────────────────────────
        self.de_emb = DEEntityEmbedding(num_entities, d)

        # ── Learnable DE blend weight ─────────────────────────────────────────
        self.de_alpha = nn.Parameter(torch.tensor(0.0))   # sigmoid(0) = 0.5

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fix_rel(self, rel: torch.Tensor) -> torch.Tensor:
        """Relation indekslarini [0, 2R-1] ga clamp qilish."""
        return rel.clamp(0, self.total_relations - 1)

    def _compute_memory(
        self,
        subjects:        torch.Tensor,       # (B,)  long
        query:           torch.Tensor,        # (B, d) float32
        snapshot_graphs: List[SnapGraph],
        device:          torch.device,
    ) -> torch.Tensor:
        """
        DaeMon uslubida path memory hisoblash.
        Barcha tensor float32 — FP16 autocast ostida xavfsiz.
        Returns: (B, N, d) float32

        H_init one-hot approach: gradient query_emb ga to'g'ri o'tadi.
        """
        B = subjects.size(0)
        N = self.num_entities
        d = self.entity_dim

        # H_init: subjects pozitsiyasiga query embedding, qolgan joy 0
        # Differentiable: one_hot (no grad) * query (has grad) → H_init has grad
        one_hot = torch.zeros(B, N, device=device, dtype=torch.float32)
        one_hot.scatter_(1, subjects.unsqueeze(1).long(), 1.0)   # (B, N) binary mask
        H_init = one_hot.unsqueeze(-1) * query.float().unsqueeze(1)  # (B, N, d) — grad ✓

        # Agar tarix bo'lmasa — faqat H_init ni qaytarish
        if not snapshot_graphs:
            return H_init

        memory   = H_init.clone()
        is_first = True

        for snap_src, snap_rel, snap_dst in snapshot_graphs:
            snap_src = snap_src.to(device)
            snap_rel = snap_rel.to(device)
            snap_dst = snap_dst.to(device)

            # Memory Passing Strategy: time-aware gate (DaeMon tawaregate)
            if is_first:
                H        = H_init.clone()
                is_first = False
            else:
                # gate: σ(W · memory)  →  (B, N, d)
                gate = torch.sigmoid(
                    self.gate_weight(memory.float())
                ).float()
                H = gate * H_init + (1.0 - gate) * memory

            # w-layer PAU (float32 ichida)
            q_f32 = query.float()
            for layer in self.pau_layers:
                H = layer(H, H_init, q_f32, snap_src, snap_rel, snap_dst, N)

            memory = H

        return memory    # (B, N, d) float32

    def _score_all(
        self,
        memory: torch.Tensor,   # (B, N, d) float32
        query:  torch.Tensor,   # (B, d)    float32
        times:  torch.Tensor,   # (B,)      long
        device: torch.device,
    ) -> torch.Tensor:
        """
        score[b, e] = memory[b,e]·query[b]  +  alpha · de(e,t_b)·query[b]
        Returns: (B, N) float32
        """
        B, N, _ = memory.shape
        all_ids  = torch.arange(N, device=device)
        alpha    = torch.sigmoid(self.de_alpha)

        # Path memory score: sum over d dimension
        path_scores = (memory * query.unsqueeze(1)).sum(-1)            # (B, N)

        # DE score — trainer timestamp-ordered: barcha batch bir xil t
        # times[0] ni ishlatamiz (torch.full ile to'ldirilgan, hammasi teng)
        t_n    = times[0].float() / float(self.num_times)
        t_exp  = t_n.expand(N)
        de_all = self.de_emb(all_ids, t_exp)                          # (N, d) — grad ✓
        de_scores = query.float() @ de_all.float().T                  # (B, N) — grad ✓

        return path_scores + alpha * de_scores                        # (B, N) f32

    # ── Forward (training) ────────────────────────────────────────────────────

    def forward(
        self,
        subjects:        torch.Tensor,
        relations:       torch.Tensor,
        objects:         torch.Tensor,
        times:           torch.Tensor,
        snapshot_graphs: List[SnapGraph],
        # Legacy compat (ignored in NEXUS):
        paths=None, path_masks=None, neg_objects=None,
        history=None,   hist_mask=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        device    = subjects.device
        rel_fixed = self._fix_rel(relations)
        query     = self.query_emb(rel_fixed).float()                  # (B, d)

        memory = self._compute_memory(subjects, query, snapshot_graphs, device)
        scores = self._score_all(memory, query, times, device)         # (B, N) f32

        # 1-N cross-entropy
        link_loss = F.cross_entropy(
            scores, objects,
            label_smoothing=self.label_smoothing,
        )

        # Orthogonal regularizer on query embeddings (DaeMon da ham bor)
        Q     = self.query_emb.weight.float()
        QTQ   = Q @ Q.T
        ortho = torch.norm(
            QTQ - torch.eye(Q.size(0), device=device), p="fro"
        )

        losses = {
            "link":     link_loss,
            "self_adv": link_loss.new_tensor(0.0),
            "ortho":    ortho,
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
        memory    = self._compute_memory(subjects, query, snapshot_graphs, device)
        return self._score_all(memory, query, times, device).clamp(-20.0, 20.0)
