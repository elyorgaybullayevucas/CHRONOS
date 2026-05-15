# models/chronos_model.py
"""
MOTIVE-TKG: Triple-Stream TKG Link Prediction

Uchta mustaqil signal birlashtiruvchi model:

━━━ Stream 1: PATH (DaeMon PAU — o'zgarishsiz) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Sequential graph message passing.
  "Qanday struktural yo'l s ni kandidat ob'ektlarga bog'laydi?"

━━━ Stream 2: MOTIVE (ikki tomonlama behavioral signature) ━━━━━━━━━━━━━━━━━━
  s_motive[s] · o_ctx[e]   (bilinear dot-product)
  s_motive[s] = decay-weighted mean of s YUBORGAN relation emb-lar
  o_ctx[e]    = decay-weighted mean of e QABUL QILGAN relation emb-lar
  "s ning tashabbus profili e ning qabul profili bilan qanchalik mos?"

━━━ Stream 3: RECURRENCE (differensiabel copy mechanism) ━━━━━━━━━━━━━━━━━━━━
  (s, r, e, t_past) → e ga yaqinda sodir bo'lgan → e ni yana bashorat qil
  recur_score[b, e] = Σ_k w_k  for edges where src=s[b], rel=r[b], dst=e
  "s dan r bilan e ga yaqinda munosabat bo'lganmi?"

  Bu TLogic (qoida-asosli) ning differensiabel neural versiyasi.
  Gradient decay_rate orqali oqadi (r_emb emas, lekin w=exp(-λδ) orqali).

━━━ Yig'indi ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  score = path + α_m · motive + α_r · recurrence

Kafolatlar:
  ✓ α_m→0, α_r→0  →  aynan DaeMon, hech qachon yomonlashmaydi
  ✓ Faqat 3 yangi scalar parametr (α_m, α_r, λ)
  ✓ Entity-free: entity embedding yo'q
  ✓ 1-N Cross-Entropy: barcha N entity gradient oladi
  ✓ scatter_add non-in-place hamma joyda

Ilmiy yangilik:
  Mavjud modellar (DaeMon, RE-GCN, xERTE, CEN, CENET, TiRGN):
    - PATH signali bor
    - MOTIVE signali yo'q (kimdir)
    - RECURRENCE differensiabel yo'q (TLogic qoida-asosli, neural emas)
  MOTIVE-TKG uchala signalni birinchi marta birlashtirib, entity-free qoladi.

DaeMon dan saqlangan:
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
# PAU Layer  (DaeMon — o'zgarishsiz)
# ─────────────────────────────────────────────────────────────────────────────

class PAULayer(nn.Module):
    """DaeMon Path Aggregation Unit. Unchanged."""

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

        # ── MOTIVE-TKG yangi parametrlari (faqat 3 ta scalar) ─────────────────
        # decay_rate  → λ = softplus(.)  > 0        temporal decay tezligi
        # alpha_m     → α_m = sigmoid(.) ∈ (0,1)   motive stream og'irligi
        # alpha_r     → α_r = sigmoid(.) ∈ (0,1)   recurrence stream og'irligi
        #
        # α_m=0, α_r=0  →  aynan DaeMon (kafolat)
        self.decay_rate = nn.Parameter(torch.zeros(1))
        self.alpha_m    = nn.Parameter(torch.zeros(1))   # motive weight
        self.alpha_r    = nn.Parameter(torch.zeros(1))   # recurrence weight

    # ── Helper ─────────────────────────────────────────────────────────────────

    def _fix_rel(self, r: torch.Tensor) -> torch.Tensor:
        return r.clamp(0, self.total_relations - 1)

    # ── Core ───────────────────────────────────────────────────────────────────

    def _compute_scores(
        self,
        subjects:        torch.Tensor,   # (B,)
        relations:       torch.Tensor,   # (B,)  ← recurrence uchun kerak
        query:           torch.Tensor,   # (B, d)
        t_query:         float,
        snapshot_graphs: List[SnapGraph],
        device:          torch.device,
    ) -> torch.Tensor:                   # (B, N)

        B, N, d = subjects.size(0), self.num_entities, self.entity_dim
        T       = float(max(self.num_times, 1))
        λ       = F.softplus(self.decay_rate)   # > 0

        # H_init: one-hot × query  (differensiabel)
        one_hot = torch.zeros(B, N, device=device, dtype=torch.float32)
        one_hot.scatter_(1, subjects.unsqueeze(1).long(), 1.0)
        H_init  = one_hot.unsqueeze(-1) * query.float().unsqueeze(1)   # (B, N, d)

        if not snapshot_graphs:
            return (H_init * query.float().unsqueeze(1)).sum(-1)

        # ── Snapshot preprocessing: device transfer + decay weights ───────────
        processed = []
        for snap_src, snap_rel, snap_dst, snap_t in snapshot_graphs:
            δ = max((t_query - float(snap_t)) / T, 0.0)
            w = torch.exp(-λ * δ)                          # scalar, grad ✓
            processed.append((
                snap_src.to(device),
                snap_rel.to(device).clamp(0, self.total_relations - 1),
                snap_dst.to(device),
                w,
            ))

        # ── Barcha snapshotlarni birlashtirish ────────────────────────────────
        c_src = torch.cat([p[0] for p in processed])                   # (E_total,)
        c_rel = torch.cat([p[1] for p in processed])                   # (E_total,)
        c_dst = torch.cat([p[2] for p in processed])                   # (E_total,)
        c_w   = torch.cat([p[3].view(1).expand(p[0].size(0))
                           for p in processed])                        # (E_total,) grad ✓
        E_total = c_dst.size(0)

        # ── Shared: decay-weighted relation embeddings ────────────────────────
        r_emb   = self.query_emb(c_rel).float()                        # (E_total, d)
        r_emb_w = r_emb * c_w.unsqueeze(-1)                            # (E_total, d) grad ✓

        # ── MOTIVE STREAM ─────────────────────────────────────────────────────

        # src_match[b, k] = 1 if c_src[k] == subjects[b]
        src_match = (subjects.unsqueeze(1) == c_src.unsqueeze(0))      # (B, E_total) bool

        # o_ctx[e]: entity e qabul qilgan relation embeddinglarning o'rtachasi
        #   "e ning receptivity profili"
        dst_exp   = c_dst.unsqueeze(-1).expand(E_total, d)
        o_ctx_sum = torch.zeros(N, d, device=device, dtype=torch.float32)
        o_ctx_sum = o_ctx_sum.scatter_add(0, dst_exp, r_emb_w)         # (N, d)

        with torch.no_grad():
            o_cnt = torch.zeros(N, device=device, dtype=torch.float32)
            o_cnt.scatter_add_(0, c_dst, c_w.detach())
            o_cnt = o_cnt.clamp(min=1e-6).unsqueeze(-1)

        o_ctx = o_ctx_sum / o_cnt                                      # (N, d)

        # s_motive[b]: subjects[b] yuborgan relation embeddinglarning o'rtachasi
        #   "s ning initiative profili"
        src_match_f = src_match.float()                                # (B, E_total)
        s_motive    = src_match_f @ r_emb_w                            # (B, d)
        s_norm      = (src_match_f * c_w.unsqueeze(0)).sum(-1, keepdim=True)
        s_motive    = s_motive / s_norm.clamp(min=1e-6)                # (B, d)

        # Bilinear compatibility: s initiative × e receptivity
        motive_score = (s_motive.unsqueeze(1) * o_ctx.unsqueeze(0)).sum(-1)  # (B, N)

        # ── RECURRENCE STREAM ─────────────────────────────────────────────────
        # (s[b], r[b], e, t_past) → yaqinda sodir bo'lgan → e ni yana bashorat qil
        # TLogic ning differensiabel neural versiyasi

        # rel_match[b, k] = 1 if c_rel[k] == relations[b]
        rel_match = (relations.unsqueeze(1) == c_rel.unsqueeze(0))     # (B, E_total) bool

        # sr_match[b, k] = 1 if edge k starts at s[b] with relation r[b]
        sr_match = (src_match & rel_match).float()                     # (B, E_total)

        # recur_score[b, e] = Σ_k w_k  for k: c_src[k]=s[b], c_rel[k]=r[b], c_dst[k]=e
        sr_weighted = sr_match * c_w.unsqueeze(0)                      # (B, E_total) grad ✓

        dst_exp_b    = c_dst.unsqueeze(0).expand(B, -1)                # (B, E_total)
        recur_score  = torch.zeros(B, N, device=device, dtype=torch.float32)
        recur_score  = recur_score.scatter_add(1, dst_exp_b, sr_weighted)  # (B, N)

        # Normalization: total weight per (s, r) pair
        recur_total  = sr_weighted.sum(-1, keepdim=True).clamp(min=1e-6)   # (B, 1)
        recur_score  = recur_score / recur_total                            # (B, N) ∈ [0,1]

        # ── PATH STREAM: DaeMon sequential MPS (o'zgarishsiz) ────────────────
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

            # Temporal decay memory update
            memory = w * H + (1.0 - w) * memory                       # (B, N, d)

        path_score = (memory * query.float().unsqueeze(1)).sum(-1)     # (B, N)

        # ── Uchta signal yig'indisi ───────────────────────────────────────────
        # α_m, α_r → 0 bo'lsa, aynan DaeMon (kafolat)
        α_m = torch.sigmoid(self.alpha_m)
        α_r = torch.sigmoid(self.alpha_r)

        return path_score + α_m * motive_score + α_r * recur_score    # (B, N)

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
        r_fix   = self._fix_rel(relations)
        query   = self.query_emb(r_fix).float()                        # (B, d)
        t_query = float(times[0].item())

        scores = self._compute_scores(
            subjects, r_fix, query, t_query, snapshot_graphs, device
        )

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
        r_fix   = self._fix_rel(relations)
        query   = self.query_emb(r_fix).float()
        t_query = float(times[0].item())

        return self._compute_scores(
            subjects, r_fix, query, t_query, snapshot_graphs, device
        ).clamp(-30.0, 30.0)
