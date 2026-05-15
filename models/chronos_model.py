# models/chronos_model.py
"""
DaeMon-Rec: Exact DaeMon + Recurrence Stream

Asosiy tuzatish:
  Oldingi modellar 1-N Cross-Entropy ishlatdi.
  DaeMon BCE + 64 self-adversarial negative ishlatadi (RotatE uslubida).

  YAGO da 10,623 entity bilan 1-N CE:
    gradient = 1/10622 har bir yolg'on entity uchun → signal juda kuchsiz

  DaeMon BCE + 64 hard negative:
    model o'zining yuqori skorli negativelarini tanlaydi (self-adversarial)
    gradient concentrated → sharp discrimination → 91%+

Arxitektura:
  PATH STREAM  = DaeMon aynan (PAU + tawaregate + sequential MPS)
  RECUR STREAM = differensiabel TLogic (CHE-TKG, 2025 tomonidan tasdiqlangan)

  score = path_score + α · recur_score

  Loss (training):  BCE + 64 self-adversarial neg (DaeMon exact)
  Loss (fallback):  1-N CE (neg_objects berilmasa)

Kafolatlar:
  ✓ α→0  →  aynan DaeMon
  ✓ neg_objects=None  →  1-N CE fallback
  ✓ entity-free
"""
from typing import Dict, List, Optional, Tuple

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
        E = src.size(0)
        device = H.device

        if E == 0:
            agg = torch.zeros(B, N, d, device=device, dtype=torch.float32)
            return self.norm(self.drop(F.relu(self.update(torch.cat([H, agg], -1)))))

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
        return self.norm(self.drop(F.relu(self.update(torch.cat([H, agg], -1)))))


# ─────────────────────────────────────────────────────────────────────────────
# DaeMon-Rec
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
        adv_temperature: float = 1.0,   # self-adversarial temp (DaeMon: 1.0)
        num_negative:    int   = 64,    # negatives per positive (DaeMon: 64)
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
        self.adv_temperature    = adv_temperature
        self.num_negative       = num_negative

        d = entity_dim
        R = num_relations * 2

        # ── DaeMon (aynan) ────────────────────────────────────────────────────
        self.query_emb   = nn.Embedding(R, d)
        self.pau_layers  = nn.ModuleList([PAULayer(d, R, dropout)
                                          for _ in range(num_pau_layers)])
        self.gate_weight = nn.Linear(d, d)

        nn.init.xavier_uniform_(self.query_emb.weight)
        nn.init.xavier_uniform_(self.gate_weight.weight)
        nn.init.zeros_(self.gate_weight.bias)

        # ── Recurrence stream ─────────────────────────────────────────────────
        self.decay_rate = nn.Parameter(torch.zeros(1))   # λ = softplus(.)
        self.alpha_r    = nn.Parameter(torch.zeros(1))   # α = sigmoid(.)

    def _fix_rel(self, r):
        return r.clamp(0, self.total_relations - 1)

    # ── Path + Recurrence scores ──────────────────────────────────────────────

    def _compute_scores(self, subjects, relations, query, t_query, snapshot_graphs, device):
        B, N, d = subjects.size(0), self.num_entities, self.entity_dim
        T = float(max(self.num_times, 1))
        λ = F.softplus(self.decay_rate)

        # H_init: one-hot × query
        one_hot = torch.zeros(B, N, device=device, dtype=torch.float32)
        one_hot.scatter_(1, subjects.unsqueeze(1).long(), 1.0)
        H_init = one_hot.unsqueeze(-1) * query.float().unsqueeze(1)

        if not snapshot_graphs:
            return (H_init * query.float().unsqueeze(1)).sum(-1)

        # Snapshot preprocessing
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

        # ── PATH STREAM: DaeMon sequential MPS (aynan) ───────────────────────
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
            memory = w * H + (1.0 - w) * memory

        path_score = (memory * query.float().unsqueeze(1)).sum(-1)     # (B, N)

        # ── RECURRENCE STREAM ─────────────────────────────────────────────────
        c_src = torch.cat([p[0] for p in processed])
        c_rel = torch.cat([p[1] for p in processed])
        c_dst = torch.cat([p[2] for p in processed])
        c_w   = torch.cat([p[3].view(1).expand(p[0].size(0)) for p in processed])

        src_match = (subjects.unsqueeze(1) == c_src.unsqueeze(0))
        rel_match = (relations.unsqueeze(1) == c_rel.unsqueeze(0))
        sr_match  = (src_match & rel_match).float()
        sr_w      = sr_match * c_w.unsqueeze(0)

        dst_exp     = c_dst.unsqueeze(0).expand(B, -1)
        recur_score = torch.zeros(B, N, device=device, dtype=torch.float32)
        recur_score = recur_score.scatter_add(1, dst_exp, sr_w)
        recur_norm  = sr_w.sum(-1, keepdim=True).clamp(min=1e-6)
        recur_score = recur_score / recur_norm

        α = torch.sigmoid(self.alpha_r)
        return path_score + α * recur_score

    # ── Self-adversarial BCE loss (DaeMon exact) ──────────────────────────────

    def _adv_bce_loss(self, scores, objects, neg_objects):
        """
        DaeMon / RotatE style self-adversarial BCE.
        scores:      (B, N)
        objects:     (B,)
        neg_objects: (B, K)
        """
        B  = scores.size(0)
        idx = torch.arange(B, device=scores.device)

        pos_score = scores[idx, objects]                              # (B,)
        neg_score = scores.gather(1, neg_objects)                     # (B, K)

        # Self-adversarial weights — detach gradient
        with torch.no_grad():
            adv_w = F.softmax(self.adv_temperature * neg_score, dim=-1)  # (B, K)

        # BCE:  -log σ(pos)  −  Σ w_k log(1 − σ(neg_k))
        # log σ(x)    = −softplus(−x)
        # log(1−σ(x)) = −softplus(x)
        pos_loss = F.softplus(-pos_score).mean()
        neg_loss = (adv_w * F.softplus(neg_score)).sum(-1).mean()

        return (pos_loss + neg_loss) / 2.0

    # ── Training ───────────────────────────────────────────────────────────────

    def forward(
        self,
        subjects, relations, objects, times, snapshot_graphs,
        paths=None, path_masks=None, neg_objects=None,
        history=None, hist_mask=None,
    ):
        device  = subjects.device
        r_fix   = self._fix_rel(relations)
        query   = self.query_emb(r_fix).float()
        t_query = float(times[0].item())

        scores = self._compute_scores(
            subjects, r_fix, query, t_query, snapshot_graphs, device
        )

        if neg_objects is not None:
            loss = self._adv_bce_loss(scores, objects, neg_objects)
        else:
            loss = F.cross_entropy(scores, objects,
                                   label_smoothing=self.label_smoothing)

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
