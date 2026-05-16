# models/chronos_model.py
"""
UniRec-TKG: Unified Temporal Graph + Recurrence

DaeMon muammosi (adabiyotda isbotlangan, TiPNN 2024):
  Sequential MPS: t1 → t2 → ... → tK
  Har bir qadam oldingi xotirani "unutadi".
  t1 dagi ma'lumot tK ga yetguncha juda zaiflashib ketadi.

UniRec-TKG yechimi:
  Barcha K snapshotni BITTA unified temporal grafga birlashtirish.
  Har bir edgening og'irligi = temporal decay(t) — yangi edge kuchli, eski kuchsiz.
  Keyin bu yagona grafda L qatlam message passing.

  Natija: DaeMon K marta PAU ishlatadi (sequential),
          UniRec-TKG L marta ishlatadi (parallel, unified).
          Uzoq o'tmish ma'lumoti bevosita ko'rinadi — yo'qolmaydi.

  + Recurrence stream: (s,r,o) takrorlanish deteksiyasi (TLogic insight)

Loss: BCE + 64 self-adversarial neg (DaeMon kabi)
Aggregation: mean + max (PNA-lite)
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SnapGraph = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]


class PAULayer(nn.Module):
    """
    Unified temporal graph uchun PAU.
    Edge weight (temporal decay) message ga kiritilgan:
      msg = (H[src] * query_rel_weight + H_init[src]) * decay_weight
    Aggregation: mean + max (PNA-lite)
    Residual: H_out = update(H, agg) + H
    """

    def __init__(self, dim: int, num_rels: int, dropout: float = 0.1):
        super().__init__()
        self.num_rels   = num_rels
        self.rel_emb    = nn.Embedding(num_rels + 1, dim, padding_idx=num_rels)
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.update     = nn.Linear(dim * 3, dim)   # H + mean + max → d
        self.norm       = nn.LayerNorm(dim)
        self.drop       = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.rel_emb.weight[:-1])

    def forward(self, H, H_init, query, src, rel, dst, N, edge_w=None):
        """
        edge_w: (E,) temporal decay weights per edge. None → uniform weight 1.
        """
        B, _, d = H.shape
        E       = src.size(0)
        device  = H.device

        if E == 0:
            agg = torch.zeros(B, N, d * 2, device=device, dtype=torch.float32)
            out = self.norm(self.drop(F.relu(self.update(
                torch.cat([H.float(), agg], dim=-1)
            ))))
            return (out + H.float())

        rel   = rel.clamp(0, self.num_rels - 1)
        w_rel = self.rel_emb(rel).float()                  # (E, d)
        w_q   = self.query_proj(query).float()             # (B, d)

        src_h  = H[:, src, :].float()                      # (B, E, d)
        init_s = H_init[:, src, :].float()                 # (B, E, d)
        msg    = src_h * (w_q.unsqueeze(1) * w_rel.unsqueeze(0)) + init_s  # (B, E, d)

        # Temporal decay weighting — uzoq o'tmish edgelari kamroq ta'sir qiladi
        if edge_w is not None:
            msg = msg * edge_w.float().view(1, E, 1)

        dst_exp = dst.view(1, E, 1).expand(B, E, d)

        # Mean aggregation
        agg_sum = torch.zeros(B, N, d, device=device, dtype=torch.float32)
        agg_sum = agg_sum.scatter_add(1, dst_exp, msg)
        with torch.no_grad():
            deg = torch.zeros(N, device=device, dtype=torch.float32)
            deg.scatter_add_(0, dst, torch.ones(E, device=device))
            deg = deg.clamp(min=1.0).view(1, N, 1)
        agg_mean = agg_sum / deg                           # (B, N, d)

        # Max aggregation
        agg_max = torch.full((B, N, d), float('-inf'), device=device, dtype=torch.float32)
        try:
            agg_max = agg_max.scatter_reduce(1, dst_exp, msg, reduce='amax')
        except (TypeError, RuntimeError):
            agg_max.scatter_(1, dst_exp, msg)
        agg_max = agg_max.masked_fill(agg_max == float('-inf'), 0.0)   # (B, N, d)

        inp   = torch.cat([H.float(), agg_mean, agg_max], dim=-1)      # (B, N, 3d)
        H_new = self.norm(self.drop(F.relu(self.update(inp))))
        return H_new + H.float()                                        # residual


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
        adv_temperature: float = 0.5,
        num_negative:    int   = 64,
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

        self.query_emb  = nn.Embedding(R, d)
        self.pau_layers = nn.ModuleList([PAULayer(d, R, dropout)
                                         for _ in range(num_pau_layers)])
        nn.init.xavier_uniform_(self.query_emb.weight)

        # Recurrence stream
        self.decay_rate = nn.Parameter(torch.zeros(1))
        self.alpha_r    = nn.Parameter(torch.zeros(1))

    def _fix_rel(self, r):
        return r.clamp(0, self.total_relations - 1)

    def _ortho_reg(self):
        W = self.query_emb.weight
        I = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
        return torch.norm(torch.mm(W, W.t()) - I, p=2)

    def _compute_scores(self, subjects, relations, query, t_query, snapshot_graphs, device):
        B, N, d = subjects.size(0), self.num_entities, self.entity_dim
        T = float(max(self.num_times, 1))
        λ = F.softplus(self.decay_rate)

        # H_init: one-hot × query
        one_hot = torch.zeros(B, N, device=device, dtype=torch.float32)
        one_hot.scatter_(1, subjects.unsqueeze(1).long(), 1.0)
        H_init  = one_hot.unsqueeze(-1) * query.float().unsqueeze(1)

        if not snapshot_graphs:
            return (H_init * query.float().unsqueeze(1)).sum(-1)

        # ── UNIFIED TEMPORAL GRAPH ────────────────────────────────────────────
        # Barcha K snapshotni bitta grafga birlashtirish.
        # Edge og'irligi = temporal decay (yangi edge → kuchli, eski → kuchsiz).
        # DaeMon: K marta PAU, sequential. Biz: L marta PAU, unified.
        src_list, rel_list, dst_list, w_list = [], [], [], []
        for snap_src, snap_rel, snap_dst, snap_t in snapshot_graphs:
            δ = max((t_query - float(snap_t)) / T, 0.0)
            w = torch.exp(-λ * δ)                                  # scalar
            n_edges = snap_src.size(0)
            src_list.append(snap_src.to(device))
            rel_list.append(snap_rel.to(device).clamp(0, self.total_relations - 1))
            dst_list.append(snap_dst.to(device))
            w_list.append(w.expand(n_edges))                       # (n_edges,)

        c_src = torch.cat(src_list)   # (E_total,)
        c_rel = torch.cat(rel_list)   # (E_total,)
        c_dst = torch.cat(dst_list)   # (E_total,)
        c_w   = torch.cat(w_list)     # (E_total,) grad ✓

        # L qatlam PAU — unified temporal grafda
        H = H_init.clone()
        for layer in self.pau_layers:
            H = layer(H, H_init, query, c_src, c_rel, c_dst, N, edge_w=c_w)

        path_score = (H * query.float().unsqueeze(1)).sum(-1)      # (B, N)

        # ── RECURRENCE STREAM ─────────────────────────────────────────────────
        src_m = (subjects.unsqueeze(1) == c_src.unsqueeze(0))
        rel_m = (relations.unsqueeze(1) == c_rel.unsqueeze(0))
        sr_w  = (src_m & rel_m).float() * c_w.unsqueeze(0)        # (B, E_total)

        dst_exp    = c_dst.unsqueeze(0).expand(B, -1)
        recur      = torch.zeros(B, N, device=device, dtype=torch.float32)
        recur      = recur.scatter_add(1, dst_exp, sr_w)
        recur_norm = sr_w.sum(-1, keepdim=True).clamp(min=1e-6)
        recur      = recur / recur_norm                            # (B, N)

        α = torch.sigmoid(self.alpha_r)
        return path_score + α * recur

    def _adv_bce_loss(self, scores, objects, neg_objects):
        B   = scores.size(0)
        idx = torch.arange(B, device=scores.device)
        pos = scores[idx, objects]
        neg = scores.gather(1, neg_objects)
        with torch.no_grad():
            adv_w = F.softmax(self.adv_temperature * neg, dim=-1)
        pos_loss = F.softplus(-pos).mean()
        neg_loss = (adv_w * F.softplus(neg)).sum(-1).mean()
        return (pos_loss + neg_loss) / 2.0 + self._ortho_reg()

    def forward(self, subjects, relations, objects, times, snapshot_graphs,
                paths=None, path_masks=None, neg_objects=None,
                history=None, hist_mask=None):
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

        return scores, {"link": loss,
                        "ortho": loss.new_tensor(0.0),
                        "self_adv": loss.new_tensor(0.0)}

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
