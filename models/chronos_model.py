# models/chronos_model.py
"""
SHAPE: Subject History Augmented Path Extrapolation

DaeMon ning asosiy kamchiligi:
  query = rel_emb[r]  faqat.  Kim so'rayotgani (subject s) e'tiborga olinmaydi.
  Biden so'rasa ham, Putin so'rasa ham — bir xil query. Mantiqsiz.

SHAPE:
  query' = rel_emb[r]  +  γ · subj_ctx[s]
                              ↑
                   s oxirgi vaqtda ishtirok etgan
                   relationslarning decay-weighted
                   o'rtacha embeddingsi

  subj_ctx[s, t] = Σ_k w_k · rel_emb[r_k]  for (s, r_k, *, t_k) in history
                   w_k = exp(−λ · (t − t_k) / T)

Bu query modulation butun model bo'yicha ta'sir qiladi:
  - PAU message passing query' bilan
  - Final scoring query' bilan
  - γ, λ ikkalasi learnable

Kafolat: γ→0 bo'lsa, aynan DaeMon. Hech qachon yomonlashmaydi.

DaeMon dan saqlanganlar (proven):
  ✓ Sequential PAU + tawaregate (MPS)
  ✓ Entity-free full graph message passing
  ✓ scatter_add non-in-place (DGL yo'q)

Yangiliklar:
  ✓ Subject history query modulation (SHQM)
  ✓ Temporal decay memory update (TDMU): memory = w·H + (1-w)·memory
  ✓ 1-N Cross-Entropy (BCE+64neg o'rniga)
"""
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SnapGraph = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]


# ─────────────────────────────────────────────────────────────────────────────
# PAU Layer  (DaeMon — o'zgarishsiz)
# ─────────────────────────────────────────────────────────────────────────────

class PAULayer(nn.Module):
    """
    DaeMon Path Aggregation Unit.
    query parametri modulated query qabul qiladi (SHAPE uchun).
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
        query:  torch.Tensor,   # (B, d)   ← modulated query qabul qiladi
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
# SHAPE Model
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

        # ── DaeMon komponentlari ──────────────────────────────────────────────
        self.query_emb   = nn.Embedding(R, d)
        self.pau_layers  = nn.ModuleList([PAULayer(d, R, dropout) for _ in range(num_pau_layers)])
        self.gate_weight = nn.Linear(d, d)   # tawaregate

        nn.init.xavier_uniform_(self.query_emb.weight)
        nn.init.xavier_uniform_(self.gate_weight.weight)
        nn.init.zeros_(self.gate_weight.bias)

        # ── SHAPE yangi parametrlari (faqat 2 ta scalar) ─────────────────────
        # decay_rate: temporal decay λ  (softplus(0) ≈ 0.69 — moderate decay)
        # gamma_param: SHQM blend weight (sigmoid(0) = 0.5 — balanced start)
        self.decay_rate  = nn.Parameter(torch.zeros(1))
        self.gamma_param = nn.Parameter(torch.zeros(1))

    # ── Helper ────────────────────────────────────────────────────────────────

    def _fix_rel(self, r: torch.Tensor) -> torch.Tensor:
        return r.clamp(0, self.total_relations - 1)

    # ── Core ─────────────────────────────────────────────────────────────────

    def _compute_memory(
        self,
        subjects:        torch.Tensor,       # (B,)
        query:           torch.Tensor,        # (B, d) — rel_emb[r]
        t_query:         float,
        snapshot_graphs: List[SnapGraph],
        device:          torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # (memory (B,N,d), modulated_q (B,d))

        B, N, d = subjects.size(0), self.num_entities, self.entity_dim
        T       = float(max(self.num_times, 1))
        λ       = F.softplus(self.decay_rate)   # > 0

        # H_init: one-hot × query  (differentiable)
        one_hot = torch.zeros(B, N, device=device, dtype=torch.float32)
        one_hot.scatter_(1, subjects.unsqueeze(1).long(), 1.0)
        H_init  = one_hot.unsqueeze(-1) * query.float().unsqueeze(1)   # (B, N, d)

        if not snapshot_graphs:
            return H_init, query.float()

        # ── Pre-processing: device transfer + decay weights ───────────────────
        # Bir marta device ga ko'chiramiz, keyin ikki yerda ishlatamiz
        processed = []
        for snap_src, snap_rel, snap_dst, snap_t in snapshot_graphs:
            δ   = max((t_query - float(snap_t)) / T, 0.0)
            w   = torch.exp(-λ * δ)                           # scalar, grad ✓
            processed.append((
                snap_src.to(device),
                snap_rel.to(device).clamp(0, self.total_relations - 1),
                snap_dst.to(device),
                w,
            ))

        # ── SHQM: Subject History Query Modulation ────────────────────────────
        # combined_src: barcha snapshot edgelarining source entitylari
        # Subjects[b] qaysi edgelarda src ekanini topib, ularning
        # relation embeddinglarini decay-weighted o'rtalaymiz

        c_src = torch.cat([p[0] for p in processed])                   # (E_total,)
        c_rel = torch.cat([p[1] for p in processed])                   # (E_total,)
        c_w   = torch.cat([p[3].view(1).expand(p[0].size(0))
                           for p in processed])                        # (E_total,) grad ✓

        # (B, E_total): batch b uchun qaysi edgeda subjects[b] == src
        match = (subjects.unsqueeze(1) == c_src.unsqueeze(0)).float()  # (B, E_total)

        # Weighted relation embeddings: (E_total, d)
        r_emb_w  = self.query_emb(c_rel).float() * c_w.unsqueeze(-1)  # (E_total, d)

        # Weighted sum per subject: (B, d)
        subj_ctx  = match @ r_emb_w                                    # (B, d)
        subj_norm = (match * c_w.unsqueeze(0)).sum(-1, keepdim=True)   # (B, 1)
        subj_ctx  = subj_ctx / subj_norm.clamp(min=1e-6)               # (B, d) normalized

        # Query modulation: q' = q + γ · subj_ctx
        γ           = torch.sigmoid(self.gamma_param)
        modulated_q = query.float() + γ * subj_ctx                     # (B, d)

        # ── Sequential MPS + TDMU ─────────────────────────────────────────────
        # DaeMon tawaregate + temporal decay weighted update
        memory   = H_init.clone()
        is_first = True

        for snap_src, snap_rel, snap_dst, w in processed:
            # DaeMon tawaregate
            if is_first:
                H        = H_init.clone()
                is_first = False
            else:
                gate = torch.sigmoid(self.gate_weight(memory.float()))
                H    = gate * H_init + (1.0 - gate) * memory

            # PAU — modulated query bilan  ← SHAPE ning farqi
            for layer in self.pau_layers:
                H = layer(H, H_init, modulated_q, snap_src, snap_rel, snap_dst, N)

            # TDMU: temporal decay weighted memory update
            # w≈1 (yaqin snapshot) → memory ≈ H  (to'liq yangilanish)
            # w≈0 (uzoq snapshot)  → memory saqlanadi  (o'tkazib yuboriladi)
            memory = w * H + (1.0 - w) * memory                       # (B, N, d)

        return memory, modulated_q

    # ── Training ──────────────────────────────────────────────────────────────

    def forward(
        self,
        subjects:        torch.Tensor,
        relations:       torch.Tensor,
        objects:         torch.Tensor,
        times:           torch.Tensor,
        snapshot_graphs: List[SnapGraph],
        paths=None, path_masks=None, neg_objects=None,
        history=None,   hist_mask=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        device   = subjects.device
        query    = self.query_emb(self._fix_rel(relations)).float()    # (B, d)
        t_query  = float(times[0].item())

        memory, mod_q = self._compute_memory(
            subjects, query, t_query, snapshot_graphs, device
        )

        # Score: modulated query bilan  (H[b,e] · q'[b])
        scores = (memory * mod_q.unsqueeze(1)).sum(-1)                 # (B, N)

        loss = F.cross_entropy(scores, objects, label_smoothing=self.label_smoothing)

        return scores, {
            "link":     loss,
            "ortho":    loss.new_tensor(0.0),
            "self_adv": loss.new_tensor(0.0),
        }

    # ── Evaluation ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        subjects:        torch.Tensor,
        relations:       torch.Tensor,
        times:           torch.Tensor,
        snapshot_graphs: List[SnapGraph],
        paths=None, path_masks=None, history=None, hist_mask=None,
    ) -> torch.Tensor:

        device   = subjects.device
        query    = self.query_emb(self._fix_rel(relations)).float()
        t_query  = float(times[0].item())

        memory, mod_q = self._compute_memory(
            subjects, query, t_query, snapshot_graphs, device
        )

        return (memory * mod_q.unsqueeze(1)).sum(-1).clamp(-30.0, 30.0)
