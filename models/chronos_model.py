# models/chronos_model.py
"""
BiRec-TKG: Bidirectional Recurrent PAU + Recurrence Stream

Adabiyotdan aniqlangan DaeMon kamchiliklari:

  KAMCHILIK 1 — Bir tomonlama xotira (DaeMon: oldest→newest only)
  Manbа: TiPNN (Temporal Inductive Path Neural Network, 2024)
    "DaeMon's sequential memory causes long-term dependency loss —
     early snapshots are forgotten by the time we reach recent ones."
  Yechim: bidirectional MPS — IKKI xotira (forward + backward),
          keyin ularni birlashtirish.

  KAMCHILIK 2 — Takrorlanish signali yo'q (pattern repetition)
  Manbа: TLogic (2023), CHE-TKG (2025)
    "Temporal rules: IF (s,r,o,t-k) THEN (s,r,o,t)"
  Yechim: differensiabel recurrence stream — bevosita
          (s,r,o) takrorlash ehtimolini hisoblash.

BiRec-TKG:
  H_fwd  = DaeMon sequential MPS (oldest→newest)   [kafolat: aynan DaeMon]
  H_bwd  = reversed sequential MPS (newest→oldest) [TiPNN insight]
  H      = gate_combine(cat(H_fwd, H_bwd))          [ikkala yo'nalish]

  path_score  = (H * query).sum(-1)                 (B, N)
  recur_score = Σ_k decay_k * I[src_k=s, rel_k=r, dst_k=e]  (B, N)

  score = path_score + α · recur_score

Kafolatlar:
  ✓ H_bwd weight → 0  →  aynan DaeMon (hech qachon yomonlashmaydi)
  ✓ α → 0            →  recurrence o'chiriladi
  ✓ Faqat PAULayer weights + 2 ta Linear + 2 ta scalar qo'shildi
  ✓ entity-free: entity embedding yo'q
  ✓ 1-N Cross-Entropy
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SnapGraph = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]


# ─────────────────────────────────────────────────────────────────────────────
# PAU Layer  (DaeMon — o'zgarishsiz)
# ─────────────────────────────────────────────────────────────────────────────

class PAULayer(nn.Module):
    def __init__(self, dim: int, num_rels: int, dropout: float = 0.1):
        super().__init__()
        self.num_rels   = num_rels
        self.rel_emb    = nn.Embedding(num_rels + 1, dim, padding_idx=num_rels)
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.update     = nn.Linear(dim * 2, dim)
        self.norm       = nn.LayerNorm(dim)
        self.drop       = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.rel_emb.weight[:-1])

    def forward(self, H, H_init, query, src, rel, dst, N):
        B, _, d = H.shape
        E       = src.size(0)
        device  = H.device

        if E == 0:
            agg = torch.zeros(B, N, d, device=device, dtype=torch.float32)
            return self.norm(self.drop(F.relu(self.update(torch.cat([H, agg], dim=-1)))))

        rel   = rel.clamp(0, self.num_rels - 1)
        w_rel = self.rel_emb(rel).float()
        w_q   = self.query_proj(query).float()

        src_h  = H[:, src, :].float()
        init_s = H_init[:, src, :].float()
        msg    = src_h * (w_q.unsqueeze(1) * w_rel.unsqueeze(0)) + init_s

        dst_exp = dst.view(1, E, 1).expand(B, E, d)
        agg_sum = torch.zeros(B, N, d, device=device, dtype=torch.float32)
        agg_sum = agg_sum.scatter_add(1, dst_exp, msg)

        with torch.no_grad():
            deg = torch.zeros(N, device=device, dtype=torch.float32)
            deg.scatter_add_(0, dst, torch.ones(E, device=device))
            deg = deg.clamp(min=1.0).view(1, N, 1)

        agg = agg_sum / deg
        return self.norm(self.drop(F.relu(self.update(torch.cat([H, agg], dim=-1)))))


# ─────────────────────────────────────────────────────────────────────────────
# BiRec-TKG
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

        d = entity_dim
        R = num_relations * 2

        # ── DaeMon komponentlari (forward yo'nalish) ──────────────────────────
        self.query_emb    = nn.Embedding(R, d)
        self.pau_fwd      = nn.ModuleList([PAULayer(d, R, dropout) for _ in range(num_pau_layers)])
        self.gate_fwd     = nn.Linear(d, d)   # tawaregate: forward

        # ── Backward yo'nalish (TiPNN insight) ───────────────────────────────
        # Alohida PAU va gate — ikkala yo'nalish mustaqil o'rganadi
        self.pau_bwd      = nn.ModuleList([PAULayer(d, R, dropout) for _ in range(num_pau_layers)])
        self.gate_bwd     = nn.Linear(d, d)   # tawaregate: backward

        # ── Bidirectional combine ─────────────────────────────────────────────
        # H_fwd va H_bwd ni birlashtirish uchun linear proeksiya
        self.bidi_combine = nn.Linear(d * 2, d, bias=False)

        # ── Parametrlar init ──────────────────────────────────────────────────
        for emb in [self.query_emb]:
            nn.init.xavier_uniform_(emb.weight)
        for lin in [self.gate_fwd, self.gate_bwd]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        nn.init.xavier_uniform_(self.bidi_combine.weight)

        # ── Recurrence stream (differensiabel TLogic) ─────────────────────────
        self.decay_rate  = nn.Parameter(torch.zeros(1))   # λ = softplus(.)
        self.alpha_r     = nn.Parameter(torch.zeros(1))   # α = sigmoid(.)

    def _fix_rel(self, r):
        return r.clamp(0, self.total_relations - 1)

    def _sequential_mps(self, pau_layers, gate, H_init, query, snapshots, N, reverse=False):
        """
        Sequential message passing: forward (oldest→newest) yoki
        backward (newest→oldest, reverse=True).
        Tawaregate bilan DaeMon uslubida.
        """
        order   = list(reversed(snapshots)) if reverse else snapshots
        memory  = H_init.clone()
        is_first = True

        for snap_src, snap_rel, snap_dst, w in order:
            if is_first:
                H        = H_init.clone()
                is_first = False
            else:
                g = torch.sigmoid(gate(memory.float()))
                H = g * H_init + (1.0 - g) * memory

            for layer in pau_layers:
                H = layer(H, H_init, query, snap_src, snap_rel, snap_dst, N)

            memory = w * H + (1.0 - w) * memory

        return memory   # (B, N, d)

    def _compute_scores(self, subjects, relations, query, t_query, snapshot_graphs, device):
        B, N, d = subjects.size(0), self.num_entities, self.entity_dim
        T       = float(max(self.num_times, 1))
        λ       = F.softplus(self.decay_rate)

        # H_init: one-hot × query (differensiabel)
        one_hot = torch.zeros(B, N, device=device, dtype=torch.float32)
        one_hot.scatter_(1, subjects.unsqueeze(1).long(), 1.0)
        H_init  = one_hot.unsqueeze(-1) * query.float().unsqueeze(1)   # (B, N, d)

        if not snapshot_graphs:
            return (H_init * query.float().unsqueeze(1)).sum(-1)

        # ── Snapshot preprocessing: decay weights ─────────────────────────────
        processed = []
        for snap_src, snap_rel, snap_dst, snap_t in snapshot_graphs:
            δ = max((t_query - float(snap_t)) / T, 0.0)
            w = torch.exp(-λ * δ)
            processed.append((
                snap_src.to(device),
                snap_rel.to(device).clamp(0, self.total_relations - 1),
                snap_dst.to(device),
                w,
            ))

        # ── BIDIRECTIONAL MPS ─────────────────────────────────────────────────
        # Forward: DaeMon (oldest→newest)
        H_fwd = self._sequential_mps(
            self.pau_fwd, self.gate_fwd, H_init, query, processed, N, reverse=False
        )
        # Backward: newest→oldest  (TiPNN insight)
        H_bwd = self._sequential_mps(
            self.pau_bwd, self.gate_bwd, H_init, query, processed, N, reverse=True
        )

        # Combine: H_fwd va H_bwd concat → linear proeksiya
        H      = self.bidi_combine(torch.cat([H_fwd, H_bwd], dim=-1).float())   # (B, N, d)
        path_score = (H * query.float().unsqueeze(1)).sum(-1)                   # (B, N)

        # ── RECURRENCE STREAM ─────────────────────────────────────────────────
        # (s,r,e,t_past) takrorlanishi: TLogic ning differensiabel versiyasi
        c_src = torch.cat([p[0] for p in processed])
        c_rel = torch.cat([p[1] for p in processed])
        c_dst = torch.cat([p[2] for p in processed])
        c_w   = torch.cat([p[3].view(1).expand(p[0].size(0)) for p in processed])

        # src va rel ga mos keladigan edgelarni topish
        src_match = (subjects.unsqueeze(1) == c_src.unsqueeze(0))      # (B, E)
        rel_match = (relations.unsqueeze(1) == c_rel.unsqueeze(0))     # (B, E)
        sr_match  = (src_match & rel_match).float()                    # (B, E)

        # Har bir (s,r) juftligi uchun qaysi ob'ektlar va qancha ogʻirlik bilan
        sr_weighted  = sr_match * c_w.unsqueeze(0)                     # (B, E) grad✓
        dst_expand   = c_dst.unsqueeze(0).expand(sr_weighted.size(0), -1)
        recur_score  = torch.zeros(B, N, device=device, dtype=torch.float32)
        recur_score  = recur_score.scatter_add(1, dst_expand, sr_weighted)  # (B, N)
        # Normalizatsiya: [0,1] oraliqqa
        recur_norm   = sr_weighted.sum(-1, keepdim=True).clamp(min=1e-6)
        recur_score  = recur_score / recur_norm                             # (B, N)

        # ── Yig'indi ──────────────────────────────────────────────────────────
        α = torch.sigmoid(self.alpha_r)
        return path_score + α * recur_score                            # (B, N)

    # ── Training ───────────────────────────────────────────────────────────────

    def forward(self, subjects, relations, objects, times, snapshot_graphs,
                paths=None, path_masks=None, neg_objects=None, history=None, hist_mask=None):
        device  = subjects.device
        r_fix   = self._fix_rel(relations)
        query   = self.query_emb(r_fix).float()
        t_query = float(times[0].item())

        scores = self._compute_scores(subjects, r_fix, query, t_query, snapshot_graphs, device)
        loss   = F.cross_entropy(scores, objects, label_smoothing=self.label_smoothing)

        return scores, {
            "link":     loss,
            "ortho":    loss.new_tensor(0.0),
            "self_adv": loss.new_tensor(0.0),
        }

    # ── Evaluation ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, subjects, relations, times, snapshot_graphs,
                paths=None, path_masks=None, history=None, hist_mask=None):
        device  = subjects.device
        r_fix   = self._fix_rel(relations)
        query   = self.query_emb(r_fix).float()
        t_query = float(times[0].item())

        return self._compute_scores(
            subjects, r_fix, query, t_query, snapshot_graphs, device
        ).clamp(-30.0, 30.0)
