# models/chronos_model.py
"""
NOVA — Neural Object-Valued Approximation for TKG Extrapolation.

Architecture:
  1. DEEntityEmbedding  — periodic + trend temporal modulation
  2. TemporalRelationEncoding — sinusoidal time + relation
  3. GRUPathEncoder     — 1-layer GRU per path + masked attention
  4. AttentiveHistoryEncoder — GRU + query-aware attention, separate projections
  5. Residual query     — query = LN(distmult + ctx_mlp([hist, path]))
       ctx_mlp zero-init → pure DistMult baseline at start
  6. Temporal scoring   — score[b,e] = query[b]·e_t_b(e), loop unique_t
  7. Loss               — BCE label-smoothing + self-adversarial

FP16 safety:
  - DE: periodic/trend computed float32, cast to e.dtype at end
  - _score_all: always float32 (query.float() + ent_emb.float())
  - masked_fill: -3e4 (not -1e9)
  - scores clamped [-10, 10]
"""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.paths import INV_OFFSET


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal encoding — always float32, dim must be even
# ─────────────────────────────────────────────────────────────────────────────

def sinusoidal_enc(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: (...) → (..., dim) float32.  dim must be even."""
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal_enc requires even dim, got {dim}")
    shape  = t.shape
    t_flat = t.float().reshape(-1, 1).clamp(0, 1e5)
    half   = dim // 2
    div    = torch.exp(
        torch.arange(half, device=t_flat.device, dtype=torch.float32)
        * -(math.log(10000.0) / half)
    )
    args = t_flat * div                                      # (N, half)
    enc  = torch.empty(t_flat.size(0), dim, device=t_flat.device, dtype=torch.float32)
    enc[:, 0::2] = torch.sin(args)
    enc[:, 1::2] = torch.cos(args)
    return enc.reshape(*shape, dim)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DE Entity Embedding — periodic + trend
# ═══════════════════════════════════════════════════════════════════════════════

class DEEntityEmbedding(nn.Module):
    """
    e_t(s) = cat(
        e_s[:D/2] ⊙ (1 + amp · sin(freq · t + phase)),   # periodic
        e_s[D/2:] + trend_s · t_norm                       # linear trend
    )

    - periodic: captures cyclic patterns (YAGO birth/death years, WIKI)
    - trend: captures long-term drift (WIKI topic evolution, ICEWS political shift)
    - trend zero-initialized → no drift at start, learned if useful
    - All modulation in float32, cast to e.dtype → FP16 safe
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
        nn.init.zeros_(self.trend.weight)   # no drift at init

    def forward(self, entities: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        """
        entities : (N,)  long
        t_norm   : (N,)  float  [0, 1]
        returns  : (N, dim)  dtype = emb.weight.dtype
        """
        e = self.emb(entities)                              # (N, dim)
        t = t_norm.float().unsqueeze(-1)                    # (N, 1) f32

        # Float32 intermediate for sin precision
        amp = torch.sigmoid(self.amp(entities).float())     # (N, mod_dim) f32
        frq = F.softplus(self.freq(entities).float())       # (N, mod_dim) f32
        phi = self.phase(entities).float()                  # (N, mod_dim) f32
        trd = self.trend(entities).float()                  # (N, mod_dim) f32

        # Periodic modulation (first half)
        mod      = amp * torch.sin(frq * t + phi)           # f32
        periodic = e[:, :self.mod_dim] * (1.0 + mod.to(e.dtype))

        # Linear trend (second half)
        trended  = e[:, self.mod_dim:] + (trd * t).to(e.dtype)

        return torch.cat([periodic, trended], dim=-1)       # (N, dim)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Temporal Relation Encoding
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalRelationEncoding(nn.Module):
    """r_t = LayerNorm(rel_emb[r] + sin_proj(t))"""

    def __init__(self, num_relations: int, dim: int, delta_dim: int = 64):
        super().__init__()
        self.delta_dim = delta_dim
        self.rel_emb   = nn.Embedding(num_relations, dim)
        self.sin_proj  = nn.Linear(delta_dim, dim)
        self.norm      = nn.LayerNorm(dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, relations: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        r  = self.rel_emb(relations)
        dt = self.sin_proj(sinusoidal_enc(times, self.delta_dim))
        return self.norm(r + dt)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GRU Path Encoder
# ═══════════════════════════════════════════════════════════════════════════════

class GRUPathEncoder(nn.Module):
    """(B, P, L) → 1-layer GRU → masked attention → (B, out_dim)"""

    def __init__(
        self,
        num_relations: int,
        rel_dim:       int,
        out_dim:       int,
        dropout:       float = 0.1,
    ):
        super().__init__()
        self.rel_emb   = nn.Embedding(num_relations, rel_dim, padding_idx=0)
        self.gru       = nn.GRU(rel_dim, out_dim, num_layers=1, batch_first=True)
        self.path_attn = nn.Linear(out_dim, 1, bias=False)
        self.norm      = nn.LayerNorm(out_dim)
        self.drop      = nn.Dropout(dropout)

    def forward(
        self,
        path_rels:  torch.Tensor,   # (B, P, L)
        path_masks: torch.Tensor,   # (B, P)  True=padding
    ) -> torch.Tensor:
        B, P, L = path_rels.shape
        rel_e     = self.rel_emb(path_rels.view(B * P, L))      # (B*P, L, rel_dim)
        _, h_n    = self.gru(rel_e)                              # (1, B*P, out_dim)
        path_vecs = self.drop(h_n.squeeze(0).view(B, P, -1))    # (B, P, out_dim)

        attn_logit = self.path_attn(path_vecs).squeeze(-1)       # (B, P)
        attn_logit = attn_logit.masked_fill(path_masks, -3e4)
        attn_w     = torch.softmax(attn_logit, dim=-1)

        valid  = (~path_masks).float()
        attn_w = attn_w * valid
        attn_w = attn_w / attn_w.sum(-1, keepdim=True).clamp(1e-8)

        has_path = valid.any(dim=-1).float().unsqueeze(-1)       # (B, 1)
        agg      = (attn_w.unsqueeze(-1) * path_vecs).sum(1)    # (B, out_dim)
        return self.norm(agg * has_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GRU + Query-Aware History Encoder
# ═══════════════════════════════════════════════════════════════════════════════

class AttentiveHistoryEncoder(nn.Module):
    """
    GRU over (rel_emb ⊕ Δt_enc) → hidden states + last state.
    Query-aware dot-product attention (DistMult signal as query).
    Gate: attn_out ↔ gru_last via separate projections (no weight sharing).
    """

    def __init__(
        self,
        num_relations: int,
        rel_dim:       int,
        hidden_dim:    int,
        out_dim:       int,
        delta_dim:     int   = 32,
        dropout:       float = 0.1,
    ):
        super().__init__()
        self.delta_dim = delta_dim
        self.rel_emb    = nn.Embedding(num_relations, rel_dim, padding_idx=0)
        self.delta_proj = nn.Linear(delta_dim, rel_dim)

        self.gru = nn.GRU(rel_dim * 2, hidden_dim, num_layers=1, batch_first=True)

        # Separate projections — key_proj for attention, h_last_proj for gate
        self.key_proj    = nn.Linear(hidden_dim, out_dim, bias=False)
        self.h_last_proj = nn.Linear(hidden_dim, out_dim, bias=False)

        self.gate_proj = nn.Sequential(nn.Linear(out_dim * 2, out_dim), nn.Sigmoid())
        self.out_proj  = nn.Linear(out_dim, out_dim)
        self.norm      = nn.LayerNorm(out_dim)
        self.drop      = nn.Dropout(dropout)
        self.scale     = out_dim ** -0.5

    def forward(
        self,
        hist_rels:    torch.Tensor,   # (B, H) long
        hist_dt:      torch.Tensor,   # (B, H) float, Δt ≥ 0
        hist_mask:    torch.Tensor,   # (B, H) bool, True=padding
        query_signal: torch.Tensor,   # (B, out_dim) DistMult
    ) -> torch.Tensor:
        B, H = hist_rels.shape

        rel_e = self.rel_emb(hist_rels)                          # (B, H, rel_dim)
        dt_e  = self.delta_proj(
            sinusoidal_enc(hist_dt.reshape(-1), self.delta_dim).reshape(B, H, -1)
        )                                                         # (B, H, rel_dim)
        x = torch.cat([rel_e, dt_e], dim=-1)                    # (B, H, 2*rel_dim)

        lengths = (~hist_mask).sum(dim=1).clamp(min=1).cpu()
        packed  = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out_packed, h_n = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out_packed, batch_first=True, total_length=H
        )                                                         # (B, H, hidden_dim)
        h_last = self.drop(h_n.squeeze(0))                      # (B, hidden_dim)

        # Attention keys (key_proj) — separate from gate projection
        hist_keys = self.key_proj(out)                           # (B, H, out_dim)
        attn_s    = (query_signal.unsqueeze(1) * hist_keys).sum(-1) * self.scale
        attn_s    = attn_s.masked_fill(hist_mask, -3e4)
        attn_w    = torch.softmax(attn_s, dim=-1)

        valid    = (~hist_mask).float()
        attn_w   = attn_w * valid
        attn_w   = attn_w / attn_w.sum(-1, keepdim=True).clamp(1e-8)
        attn_out = (attn_w.unsqueeze(-1) * hist_keys).sum(1)    # (B, out_dim)

        # Last state (h_last_proj) — separate linear
        h_proj = self.h_last_proj(h_last)                       # (B, out_dim)

        gate  = self.gate_proj(torch.cat([attn_out, h_proj], dim=-1))
        fused = gate * attn_out + (1 - gate) * h_proj
        out_v = self.out_proj(self.drop(fused))

        has_hist = valid.any(dim=-1).float().unsqueeze(-1)
        return self.norm(out_v * has_hist)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. NOVA — main model
# ═══════════════════════════════════════════════════════════════════════════════

class CHRONOSModel(nn.Module):
    """
    NOVA: DistMult + DE(periodic+trend) + GRU-Path + GRU-QA-History.

    Residual query:
      query = LN(distmult + ctx_mlp([hist_sig, path_sig]))
      ctx_mlp last layer zero-init → starts as pure DistMult, learns corrections.

    score[b,e] = query[b] · e_t_b(e)  — temporally correct, no B×E×D.
    """

    def __init__(
        self,
        num_entities:    int,
        num_relations:   int,
        num_times:       int,
        entity_dim:      int   = 256,
        relation_dim:    int   = 256,
        hidden_dim:      int   = 512,
        delta_dim:       int   = 64,
        # backward compat (unused)
        num_heads:       int   = 8,
        num_layers:      int   = 2,
        ffn_dim:         int   = 1024,
        num_patterns:    int   = 128,
        num_trtm_bins:   int   = 32,
        num_negative:    int   = 256,
        dropout:         float = 0.1,
        label_smoothing: float = 0.1,
        w_direct:        float = 0.0,
        w_pcl:           float = 0.0,
        use_history:     bool  = True,
        max_history:     int   = 32,
    ):
        super().__init__()
        self.num_entities       = num_entities
        self.num_base_relations = num_relations
        self.total_relations    = num_relations * 2
        self.num_times          = max(num_times, 1)
        self.entity_dim         = entity_dim
        self.use_history        = use_history
        self.label_smoothing    = label_smoothing

        # 1. DE embedding (periodic + trend)
        self.de_emb = DEEntityEmbedding(num_entities, entity_dim)

        # 2. Temporal relation encoding
        self.tre = TemporalRelationEncoding(
            self.total_relations, entity_dim, delta_dim
        )

        # 3. GRU path encoder  (rel_dim = entity_dim // 2)
        self.path_enc = GRUPathEncoder(
            num_relations = self.total_relations,
            rel_dim       = entity_dim // 2,
            out_dim       = entity_dim,
            dropout       = dropout,
        )

        # 4. GRU + QA history encoder
        if use_history:
            self.hist_enc = AttentiveHistoryEncoder(
                num_relations = self.total_relations,
                rel_dim       = entity_dim // 2,
                hidden_dim    = hidden_dim // 2,
                out_dim       = entity_dim,
                delta_dim     = delta_dim // 2,
                dropout       = dropout,
            )

        # 5. Context MLP: [hist_sig, path_sig] → additive correction to distmult
        #    Input is always 2*entity_dim (hist_sig=zeros when no history)
        self.ctx_mlp = nn.Sequential(
            nn.Linear(entity_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, entity_dim),
        )
        # Zero-init last layer → ctx=0 at init → pure DistMult baseline
        nn.init.zeros_(self.ctx_mlp[-1].weight)
        nn.init.zeros_(self.ctx_mlp[-1].bias)

        # 6. Query normalization
        self.query_norm = nn.LayerNorm(entity_dim)

        # 7. Self-adversarial temperature per relation
        self.rel_temp = nn.Embedding(self.total_relations, 1)
        nn.init.constant_(self.rel_temp.weight, 1.0)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fix_rel(self, rel: torch.Tensor) -> torch.Tensor:
        """INV_OFFSET → num_base_relations offset, clamp to valid range."""
        inv = rel >= INV_OFFSET
        if inv.any():
            rel = rel.clone()
            rel[inv] = (rel[inv] - INV_OFFSET) + self.num_base_relations
        return rel.clamp(0, self.total_relations - 1)

    def _t_norm(self, times: torch.Tensor) -> torch.Tensor:
        return times.float() / float(self.num_times)

    # ── Query builder ─────────────────────────────────────────────────────────

    def _build_query(
        self,
        subjects:   torch.Tensor,           # (B,)
        relations:  torch.Tensor,           # (B,) already _fix_rel'd
        times:      torch.Tensor,           # (B,)
        paths:      torch.Tensor,           # (B, P, L, 3)
        path_masks: torch.Tensor,           # (B, P)  True=padding
        history:    Optional[torch.Tensor], # (B, H, 3)
        hist_mask:  Optional[torch.Tensor], # (B, H)  True=padding
    ) -> torch.Tensor:
        """Returns query (B, entity_dim)."""
        B   = subjects.size(0)
        t_n = self._t_norm(times)

        s_t      = self.de_emb(subjects, t_n)              # (B, D)
        r_t      = self.tre(relations, times)              # (B, D)
        distmult = s_t * r_t                               # (B, D) — base signal

        path_rels = self._fix_rel(paths[:, :, :, 1])      # (B, P, L)
        path_sig  = self.path_enc(path_rels, path_masks)  # (B, D)

        if self.use_history and history is not None:
            hist_rels = self._fix_rel(history[:, :, 1])   # (B, H)
            hist_dt   = (times.unsqueeze(-1) - history[:, :, 2]).float().clamp(0)
            hist_sig  = self.hist_enc(hist_rels, hist_dt, hist_mask, distmult)
        else:
            hist_sig = distmult.new_zeros(B, self.entity_dim)

        # Residual: distmult preserved, ctx_mlp adds learned correction
        # When hist=zeros and path=zeros: ctx≈0 (zero-init) → query≈LN(distmult)
        ctx   = self.ctx_mlp(torch.cat([hist_sig, path_sig], dim=-1))
        query = self.query_norm(distmult + ctx)
        return query

    # ── Temporal all-entity scoring (predict only) ────────────────────────────

    def _score_all(
        self, query: torch.Tensor, times: torch.Tensor
    ) -> torch.Tensor:
        """
        score[b, e] = query[b] · e_{t_b}(e)

        Loop over unique time steps: per step, (E, D) DE once → (b_i, E) matmul.
        No B×E×D expansion. All float32 for FP16 safety.
        """
        B      = query.size(0)
        E      = self.num_entities
        device = query.device

        unique_t, t_inv = torch.unique(times, return_inverse=True)
        all_ids = torch.arange(E, device=device)
        scores  = torch.zeros(B, E, device=device, dtype=torch.float32)
        q32     = query.float()

        for i, t_val in enumerate(unique_t):
            mask    = t_inv == i                             # (B,) bool
            t_n     = t_val.float() / float(self.num_times)
            t_exp   = t_n.unsqueeze(0).expand(E)            # (E,) same value
            ent_emb = self.de_emb(all_ids, t_exp).float()  # (E, D) f32
            scores[mask] = (q32[mask] @ ent_emb.T).float()  # (b_i, E) — force f32

        return scores.clamp(-10, 10)

    # ── Forward (training) ────────────────────────────────────────────────────

    def forward(
        self,
        subjects:    torch.Tensor,
        relations:   torch.Tensor,
        objects:     torch.Tensor,
        times:       torch.Tensor,
        paths:       torch.Tensor,
        path_masks:  torch.Tensor,
        neg_objects: torch.Tensor,
        history:     Optional[torch.Tensor] = None,
        hist_mask:   Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        B         = subjects.size(0)
        rel_fixed = self._fix_rel(relations)
        t_norm    = self._t_norm(times)

        query = self._build_query(
            subjects, rel_fixed, times, paths, path_masks, history, hist_mask
        )

        # Temporal scoring: DE applied to object side too
        pos_emb   = self.de_emb(objects, t_norm)                       # (B, D)
        pos_score = (query * pos_emb).sum(-1)                          # (B,)

        has_neg = neg_objects.size(1) > 0
        if has_neg:
            N       = neg_objects.size(1)
            t_exp   = t_norm.unsqueeze(1).expand(B, N).reshape(-1)    # (B*N,)
            neg_emb = self.de_emb(neg_objects.reshape(-1), t_exp)     # (B*N, D)
            neg_score = (
                query.unsqueeze(1) * neg_emb.view(B, N, -1)
            ).sum(-1)                                                   # (B, N)
        else:
            neg_score = query.new_zeros(B, 1)

        # BCE with label smoothing
        sm = self.label_smoothing

        def bce_ls(logit: torch.Tensor, label: float) -> torch.Tensor:
            tgt = torch.full_like(logit, label * (1 - sm) + sm * 0.5)
            return F.binary_cross_entropy_with_logits(logit, tgt)

        link_loss = bce_ls(pos_score, 1.0) + bce_ls(neg_score, 0.0)

        # Self-adversarial negative sampling
        if has_neg:
            with torch.no_grad():
                temp  = self.rel_temp(rel_fixed).squeeze(-1).abs().clamp(0.5, 10)
                adv_w = torch.softmax(
                    neg_score.detach() * temp.unsqueeze(-1), dim=1
                )
            sa_loss = -(adv_w * F.logsigmoid(-neg_score)).sum(1).mean()
        else:
            sa_loss = query.new_tensor(0.0)

        losses = {
            "link":      link_loss,
            "self_adv":  sa_loss,
            "pcl":       query.new_tensor(0.0),
            "ortho_reg": query.new_tensor(0.0),
        }
        return pos_score, losses

    # ── Predict (evaluation) ──────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        subjects:   torch.Tensor,
        relations:  torch.Tensor,
        times:      torch.Tensor,
        paths:      torch.Tensor,
        path_masks: torch.Tensor,
        history:    Optional[torch.Tensor] = None,
        hist_mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        rel_fixed = self._fix_rel(relations)
        query     = self._build_query(
            subjects, rel_fixed, times, paths, path_masks, history, hist_mask
        )
        return self._score_all(query, times)
