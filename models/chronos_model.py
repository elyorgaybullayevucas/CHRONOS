# models/chronos_model.py
"""
STELLAR — Structured TEmporal Link learning with Attentive Local Reasoning.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Komponentlar:

  1. DE Embedding   — Diachronic Entity Embedding (YAGO/WIKI uchun asosiy)
                     e_t(s) = [e_s[:D/2] ⊙ (1 + A_s·sin(F_s·t+P_s)), e_s[D/2:]]

  2. TRE            — Temporal Relation Encoding
                     r_t = LayerNorm(rel_emb + dt_proj(sinenc(t)))

  3. DistMult Core  — Asosiy signal: h = e_t(s) ⊙ r_t
                     Samarali, isbotlangan (TComplEx, TNTComplEx)

  4. GRU Path Enc   — Har yo'l uchun 1-layer GRU + attention paths bo'yicha
                     BFS emas, mean emas → tartib saqlanadi

  5. GRU+QA History — GRU bilan history encode, keyin DistMult signali bilan
                     query-aware dot-product attention (DaeMon GTM g'oyasi)

  6. Query MLP      — [h ⊕ hist ⊕ path] → LayerNorm → entity_dim

  7. Scoring        — query · ent_emb.T  → clamp(-10,10)
                     BITTA matmul. Hech qanday (B×E×D) kengaytirish yo'q.

  8. Losses         — BCE label smoothing + Self-adversarial

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Nima YO'Q (muammo sabablari):
  - (B × E × 3D) expansion    → NaN, xotira
  - Transformer path encoder  → sekin
  - masked_fill(-1e9)         → FP16 overflow
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.paths import INV_OFFSET

# ─────────────────────────────────────────────────────────────────────────────
# Yordamchi: sinusoidal Δt encoding
# ─────────────────────────────────────────────────────────────────────────────

def sinusoidal_enc(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: (...) float tensor
    returns: (..., dim)
    """
    shape = t.shape
    t = t.float().reshape(-1, 1).clamp(0, 1e5)      # (N, 1)
    div = torch.exp(
        torch.arange(0, dim, 2, device=t.device, dtype=torch.float32)
        * -(math.log(10000.0) / dim)
    )                                                # (dim//2,)
    args = t * div                                   # (N, dim//2)
    enc  = torch.zeros(t.size(0), dim, device=t.device, dtype=torch.float32)
    enc[:, 0::2] = torch.sin(args)
    enc[:, 1::2] = torch.cos(args)
    return enc.reshape(*shape, dim)                  # (..., dim)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DE Entity Embedding — Diachronic
# ═══════════════════════════════════════════════════════════════════════════════

class DEEntityEmbedding(nn.Module):
    """
    Har bir entity uchun statik + vaqtiy (sinusoidal) modulyatsiya.

    e_t(s) = cat(
        e_s[:mod_dim] ⊙ (1 + amp_s · sin(freq_s · t_norm + phase_s)),
        e_s[mod_dim:]
    )

    mod_dim = dim // 2  → birinchi yarmida temporal, qolgani statik.

    Bu YAGO (tug'ilgan yil, o'lgan yil, turmush qurilgan yil) va
    WIKI uchun eng muhim komponent.
    """

    def __init__(self, num_entities: int, dim: int):
        super().__init__()
        self.dim     = dim
        self.mod_dim = dim // 2

        self.emb   = nn.Embedding(num_entities, dim)
        self.amp   = nn.Embedding(num_entities, self.mod_dim)
        self.freq  = nn.Embedding(num_entities, self.mod_dim)
        self.phase = nn.Embedding(num_entities, self.mod_dim)

        nn.init.xavier_uniform_(self.emb.weight)
        nn.init.constant_(self.amp.weight,   0.1)
        nn.init.uniform_(self.freq.weight,   0.01, 2.0)
        nn.init.uniform_(self.phase.weight,  0.0,  math.pi)

    def forward(self, entities: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        """
        entities : (B,)
        t_norm   : (B,)  — t / num_times, [0, 1]
        returns  : (B, dim)
        """
        e   = self.emb(entities)                             # (B, dim)
        amp = torch.sigmoid(self.amp(entities))              # (B, mod_dim)
        frq = F.softplus(self.freq(entities))                # (B, mod_dim) pozitiv
        phi = self.phase(entities)                           # (B, mod_dim)

        t   = t_norm.float().unsqueeze(-1)                   # (B, 1)
        mod = amp * torch.sin(frq * t + phi)                 # (B, mod_dim)

        mod_part    = e[:, :self.mod_dim] * (1.0 + mod)
        static_part = e[:, self.mod_dim:]
        return torch.cat([mod_part, static_part], dim=-1)   # (B, dim)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Temporal Relation Encoding (TRE)
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalRelationEncoding(nn.Module):
    """
    r_t = LayerNorm(rel_emb[r] + dt_proj(sinenc(t)))

    Har bir relatsiya vaqtiy kontekst bilan boyitiladi.
    YAGO da "married" faqat ma'lum davrlarda faol — bu shu signalni ushlaydi.
    """

    def __init__(self, num_relations: int, dim: int, delta_dim: int = 64):
        super().__init__()
        self.delta_dim  = delta_dim
        self.rel_emb    = nn.Embedding(num_relations, dim)
        self.delta_proj = nn.Linear(delta_dim, dim)
        self.norm       = nn.LayerNorm(dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, relations: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """relations, times: (B,) → (B, dim)."""
        r   = self.rel_emb(relations)
        dt  = self.delta_proj(sinusoidal_enc(times.float(), self.delta_dim))
        return self.norm(r + dt)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GRU Path Encoder
# ═══════════════════════════════════════════════════════════════════════════════

class GRUPathEncoder(nn.Module):
    """
    Har yo'l uchun 1-layer GRU, keyin paths bo'yicha attention.

    Mean pooling dan yaxshiroq: GRU yo'l ichidagi HOP TARTIBINI saqlaydi.
    Transformer dan tezroq: faqat 1-layer GRU.

    path_rels: (B, P, L) — har yo'ldagi relatsiya indekslari
    """

    def __init__(
        self,
        num_relations: int,
        rel_dim:       int,
        out_dim:       int,
        dropout:       float = 0.1,
    ):
        super().__init__()
        self.rel_emb = nn.Embedding(num_relations, rel_dim, padding_idx=0)
        self.gru     = nn.GRU(rel_dim, out_dim, num_layers=1, batch_first=True)
        self.path_attn = nn.Linear(out_dim, 1, bias=False)
        self.norm    = nn.LayerNorm(out_dim)
        self.drop    = nn.Dropout(dropout)

    def forward(
        self,
        path_rels:  torch.Tensor,   # (B, P, L)
        path_masks: torch.Tensor,   # (B, P)  True=padding
    ) -> torch.Tensor:
        B, P, L = path_rels.shape

        # Barcha yo'llarni bir batchga joylaymiz
        rel_e = self.rel_emb(path_rels.view(B * P, L))          # (B*P, L, rel_dim)

        # GRU: oxirgi hidden state
        _, h_n = self.gru(rel_e)                                 # h_n: (1, B*P, out_dim)
        path_vecs = h_n.squeeze(0).view(B, P, -1)               # (B, P, out_dim)
        path_vecs = self.drop(path_vecs)

        # Paths bo'yicha attention
        attn_logit = self.path_attn(path_vecs).squeeze(-1)      # (B, P)
        attn_logit = attn_logit.masked_fill(path_masks, -3e4)
        attn_w     = torch.softmax(attn_logit, dim=-1)

        # Padding yo'llarni sifirga
        valid = (~path_masks).float()
        attn_w = attn_w * valid
        denom  = attn_w.sum(-1, keepdim=True).clamp(min=1e-8)
        attn_w = attn_w / denom

        has_path = valid.any(dim=-1).float().unsqueeze(-1)       # (B, 1)
        agg      = (attn_w.unsqueeze(-1) * path_vecs).sum(1)    # (B, out_dim)
        return self.norm(agg * has_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GRU + Query-Aware History Encoder
# ═══════════════════════════════════════════════════════════════════════════════

class AttentiveHistoryEncoder(nn.Module):
    """
    DaeMon GTM g'oyasi: history dan query ga mos elementlarni tanlash.

    Qadamlar:
      1. GRU → barcha history elementlari uchun hidden state
      2. Dot-product attention: distmult_signal × hist_hidden
         Bu query-aware: faqat joriy (s, r, t) ga mos tarixiy faktlar tanlanadi
      3. Gate: weighted history ↔ GRU last state
      4. Chiqish: entity_dim

    DaeMon dan farqi: Transformer yo'q → 5-10x tezroq.
    DaeMon dan o'xshashligi: query-aware selection → sifat saqlanadi.
    """

    def __init__(
        self,
        num_relations: int,
        rel_dim:       int,
        hidden_dim:    int,
        out_dim:       int,
        delta_dim:     int = 32,
        dropout:       float = 0.1,
    ):
        super().__init__()
        self.delta_dim  = delta_dim
        self.rel_emb    = nn.Embedding(num_relations, rel_dim, padding_idx=0)
        self.delta_proj = nn.Linear(delta_dim, rel_dim)

        # GRU
        self.gru = nn.GRU(
            rel_dim * 2, hidden_dim,
            num_layers=1, batch_first=True,
        )
        # Hidden → out_dim (attention key uchun)
        self.hist_key = nn.Linear(hidden_dim, out_dim, bias=False)

        # Gate: (distmult_attn, gru_last) → out
        self.gate_proj = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm     = nn.LayerNorm(out_dim)
        self.drop     = nn.Dropout(dropout)
        self.scale    = out_dim ** -0.5

    def forward(
        self,
        hist_rels:    torch.Tensor,   # (B, H)
        hist_dt:      torch.Tensor,   # (B, H)   Δt = t_query - t_hist
        hist_mask:    torch.Tensor,   # (B, H)   True = padding
        query_signal: torch.Tensor,  # (B, out_dim) — DistMult signali
    ) -> torch.Tensor:
        """Returns: (B, out_dim)."""
        B, H = hist_rels.shape

        rel_e   = self.rel_emb(hist_rels)                       # (B, H, rel_dim)
        dt_e    = self.delta_proj(
            sinusoidal_enc(hist_dt.reshape(-1), self.delta_dim)
            .reshape(B, H, -1)
        )                                                        # (B, H, rel_dim)
        x       = torch.cat([rel_e, dt_e], dim=-1)             # (B, H, 2*rel_dim)

        # GRU (pack for efficiency)
        lengths = (~hist_mask).sum(dim=1).clamp(min=1).cpu()
        packed  = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out_packed, h_n = self.gru(packed)
        out, _  = nn.utils.rnn.pad_packed_sequence(
            out_packed, batch_first=True, total_length=H
        )                                                        # (B, H, hidden)
        h_last  = self.drop(h_n.squeeze(0))                    # (B, hidden)

        # History keys
        hist_keys = self.hist_key(out)                          # (B, H, out_dim)

        # Query-aware dot-product attention
        q       = query_signal.unsqueeze(1)                     # (B, 1, out_dim)
        attn_s  = (q * hist_keys).sum(-1) * self.scale         # (B, H)
        attn_s  = attn_s.masked_fill(hist_mask, -3e4)
        attn_w  = torch.softmax(attn_s, dim=-1)

        # Padding yo'llarni sifirga
        valid   = (~hist_mask).float()
        attn_w  = attn_w * valid
        denom   = attn_w.sum(-1, keepdim=True).clamp(1e-8)
        attn_w  = attn_w / denom
        attn_out = (attn_w.unsqueeze(-1) * hist_keys).sum(1)   # (B, out_dim)

        # GRU last → out_dim
        h_proj  = self.hist_key(h_last)                        # (B, out_dim)

        # Gate: attn + gru_last
        gate    = self.gate_proj(torch.cat([attn_out, h_proj], dim=-1))
        fused   = gate * attn_out + (1 - gate) * h_proj       # (B, out_dim)
        out_v   = self.out_proj(self.drop(fused))              # (B, out_dim)

        # Tarix bo'lmasa — nol
        has_hist = valid.any(dim=-1).float().unsqueeze(-1)
        return self.norm(out_v * has_hist)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. STELLAR — asosiy model
# ═══════════════════════════════════════════════════════════════════════════════

class CHRONOSModel(nn.Module):
    """
    STELLAR: DistMult + DE + GRU History (QA-attn) + GRU Path.

    Scoring: query · ent_emb.T  (bitta matmul, tez, stable).
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
        self.hidden_dim         = hidden_dim
        self.use_history        = use_history
        self.label_smoothing    = label_smoothing

        # ── 1. DE Entity Embedding ────────────────────────────────────────────
        self.de_emb = DEEntityEmbedding(num_entities, entity_dim)

        # ── 2. Temporal Relation Encoding ─────────────────────────────────────
        self.tre = TemporalRelationEncoding(
            self.total_relations, entity_dim, delta_dim
        )

        # ── 3. GRU Path Encoder ───────────────────────────────────────────────
        # Relatsiya dim = entity_dim // 2 (path uchun kichikroq yetarli)
        self.path_enc = GRUPathEncoder(
            num_relations = self.total_relations,
            rel_dim       = entity_dim // 2,
            out_dim       = entity_dim,
            dropout       = dropout,
        )

        # ── 4. GRU + QA History Encoder ───────────────────────────────────────
        if use_history:
            self.hist_enc = AttentiveHistoryEncoder(
                num_relations = self.total_relations,
                rel_dim       = entity_dim // 2,
                hidden_dim    = hidden_dim // 2,
                out_dim       = entity_dim,
                delta_dim     = delta_dim // 2,
                dropout       = dropout,
            )

        # ── 5. Query MLP: [distmult ⊕ hist ⊕ path] → entity_dim ──────────────
        n_signals = 3 if use_history else 2
        self.query_mlp = nn.Sequential(
            nn.Linear(entity_dim * n_signals, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, entity_dim),
            nn.LayerNorm(entity_dim),
        )

        # ── 6. Self-adversarial temperature ───────────────────────────────────
        self.rel_temp = nn.Embedding(self.total_relations, 1)
        nn.init.constant_(self.rel_temp.weight, 1.0)

    # ── Yordamchi ─────────────────────────────────────────────────────────────

    def _fix_rel(self, rel: torch.Tensor) -> torch.Tensor:
        """r+INV_OFFSET → r+num_base_relations  (clamp qilinadi)."""
        inv = rel >= INV_OFFSET
        if inv.any():
            rel = rel.clone()
            rel[inv] = (rel[inv] - INV_OFFSET) + self.num_base_relations
        return rel.clamp(0, self.total_relations - 1)

    def _t_norm(self, times: torch.Tensor) -> torch.Tensor:
        return times.float() / float(self.num_times)

    def _build_query(
        self,
        subjects:   torch.Tensor,
        relations:  torch.Tensor,   # already fixed
        times:      torch.Tensor,
        paths:      torch.Tensor,   # (B, P, L, 3)
        path_masks: torch.Tensor,   # (B, P)
        history:    Optional[torch.Tensor],   # (B, H, 3)
        hist_mask:  Optional[torch.Tensor],   # (B, H)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          query      : (B, entity_dim)
          distmult_h : (B, entity_dim)  — DistMult signali (history attn uchun)
        """
        B = subjects.size(0)
        t_n = self._t_norm(times)

        # DE entity embedding
        s_t = self.de_emb(subjects, t_n)                        # (B, D)
        # Temporal relation encoding
        r_t = self.tre(relations, times)                        # (B, D)
        # DistMult core
        distmult = s_t * r_t                                    # (B, D)

        # Path signal
        path_rels = self._fix_rel(paths[:, :, :, 1])           # (B, P, L)
        path_sig  = self.path_enc(path_rels, path_masks)       # (B, D)

        # History signal (query-aware)
        if self.use_history and history is not None:
            hist_rels = self._fix_rel(history[:, :, 1])        # (B, H)
            hist_dt   = (times.unsqueeze(-1) - history[:, :, 2]).float().clamp(0)
            hist_sig  = self.hist_enc(
                hist_rels, hist_dt, hist_mask, distmult
            )                                                   # (B, D)
        else:
            hist_sig = distmult.new_zeros(B, self.entity_dim)

        # Query assembly
        if self.use_history:
            cat_in = torch.cat([distmult, hist_sig, path_sig], dim=-1)
        else:
            cat_in = torch.cat([distmult, path_sig], dim=-1)

        query = self.query_mlp(cat_in)                          # (B, D)
        return query, distmult

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

        query, _ = self._build_query(
            subjects, rel_fixed, times, paths, path_masks, history, hist_mask
        )

        # Scoring: query · ent_emb.T
        # Static embeddings for all entities — efficient, no B×E×D expansion.
        # Temporal context is encoded in query via DE + TRE.
        all_ent = self.de_emb.emb.weight                       # (E, D)
        scores  = (query @ all_ent.T).clamp(-10, 10)           # (B, E)

        # ── Link loss (label smoothing) ───────────────────────────────────────
        pos_score = scores[torch.arange(B), objects]           # (B,)
        has_neg   = neg_objects.size(1) > 0
        neg_score = scores.gather(1, neg_objects) if has_neg \
                    else scores.new_zeros(B, 1)

        sm = self.label_smoothing
        def bce_ls(logit: torch.Tensor, label: float) -> torch.Tensor:
            t = torch.full_like(logit, label * (1 - sm) + sm * 0.5)
            return F.binary_cross_entropy_with_logits(logit, t)

        link_loss = bce_ls(pos_score, 1.0) + bce_ls(neg_score, 0.0)

        # ── Self-adversarial loss ─────────────────────────────────────────────
        if has_neg:
            with torch.no_grad():
                temp  = self.rel_temp(rel_fixed).squeeze(-1).abs().clamp(0.5, 10)
                adv_w = torch.softmax(
                    neg_score.detach() * temp.unsqueeze(-1), dim=1
                )
            sa_loss = -(adv_w * F.logsigmoid(-neg_score)).sum(1).mean()
        else:
            sa_loss = scores.new_tensor(0.0)

        losses = {
            "link":      link_loss,
            "self_adv":  sa_loss,
            "pcl":       scores.new_tensor(0.0),
            "ortho_reg": scores.new_tensor(0.0),
        }
        return scores, losses

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
        query, _  = self._build_query(
            subjects, rel_fixed, times, paths, path_masks, history, hist_mask
        )
        return (query @ self.de_emb.emb.weight.T).clamp(-10, 10)
