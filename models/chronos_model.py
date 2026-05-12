# models/chronos_model.py
"""
AURORA — Adaptive Unified Representation for Ordered Reasoning in time-Aware graphs.

Arxitektura (sodda, tez, ishonchli):

  1. DE Embedding   — entity × time sine modulation (YAGO/WIKI uchun muhim)
  2. Temporal Rel   — relation + sinusoidal(Δt) encoding
  3. GRU History    — sorted history → GRU → gated signal
  4. Path Encoding  — mean(rel_embs along path) + attention across paths
  5. Query MLP      — [s_t | r_t | hist | path] → query vector
  6. Scoring        — query · ent_emb.T  (B×E matmul, NO MLP over entities)
  7. Self-Adv Loss  — link BCE + self-adversarial negatives

MUAMMO YO'Q:
  - (B × E × 3D) expansion yo'q → NaN yo'q, xotira to'lib ketmaydi
  - FP16 safe: clamp(-10,10), masked_fill(-3e4)
  - GRU: Transformer dan 5-10x tezroq
"""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.paths import INV_OFFSET


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DE — Diachronic Entity Embedding
# ══════════════════════════════════════════════════════════════════════════════

class DiachronicEmbedding(nn.Module):
    """
    DE-SimplE dan ilhomlangan vaqtiy entity embedding.

    e_t(s) = cat(
        e_s[:D//2] * (1 + amp_s * sin(freq_s * t_norm + phase_s)),
        e_s[D//2:]
    )

    Birinchi yarmida vaqtiy modulyatsiya, ikkinchi yarmi statik.
    Bu YAGO/WIKI uchun juda muhim: "born_in 1970" kabi relatsiyalar
    faqat ma'lum yillarda faol.

    Parametr soni: E × (D + D//2 × 3) — mos hajmda.
    """

    def __init__(self, num_entities: int, dim: int):
        super().__init__()
        self.dim      = dim
        self.mod_dim  = dim // 2
        self.stat_dim = dim - dim // 2

        self.emb   = nn.Embedding(num_entities, dim)
        self.freq  = nn.Embedding(num_entities, self.mod_dim)
        self.phase = nn.Embedding(num_entities, self.mod_dim)
        self.amp   = nn.Embedding(num_entities, self.mod_dim)

        nn.init.xavier_uniform_(self.emb.weight)
        nn.init.uniform_(self.freq.weight,  0.01, 1.0)
        nn.init.uniform_(self.phase.weight, 0.0,  math.pi)
        nn.init.constant_(self.amp.weight,  0.1)

    def forward(
        self,
        entities: torch.Tensor,   # (B,) yoki (E,)
        t_norm:   torch.Tensor,   # (B,) yoki scalar, 0..1 oralig'ida
    ) -> torch.Tensor:
        e    = self.emb(entities)                                 # (..., D)
        freq = self.freq(entities)                                # (..., D//2)
        phi  = self.phase(entities)
        amp  = torch.sigmoid(self.amp(entities))                  # 0..1

        t = t_norm.float()
        if t.dim() < e.dim() - 1:
            t = t.unsqueeze(-1)                                   # broadcast

        mod  = amp * torch.sin(freq * t + phi)                   # (..., D//2)
        mod_part    = e[..., :self.mod_dim] * (1.0 + mod)
        static_part = e[..., self.mod_dim:]
        return torch.cat([mod_part, static_part], dim=-1)        # (..., D)

    def all_embeddings(
        self,
        t_norm: torch.Tensor,   # (B,) — har bir query uchun vaqt
        device: torch.device,
    ) -> torch.Tensor:
        """
        Barcha entity lar uchun o'rtacha vaqt modulyatsiyasi.
        Scoring uchun: query · ent_emb.T
        t_norm o'rniga faqat statik emb ishlatiladi (tezroq va yetarli).
        """
        return self.emb.weight                                    # (E, D)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Temporal Relation Encoding
# ══════════════════════════════════════════════════════════════════════════════

class TemporalRelationEncoding(nn.Module):
    """
    r_t = LayerNorm(rel_emb[r] + proj(sin_enc(t)))

    Har bir relatsiya vaqtiy kontekst bilan boyitiladi.
    """

    def __init__(self, num_relations: int, dim: int, delta_dim: int = 64):
        super().__init__()
        self.rel_emb    = nn.Embedding(num_relations, dim)
        self.delta_dim  = delta_dim
        self.delta_proj = nn.Linear(delta_dim, dim)
        self.norm       = nn.LayerNorm(dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def _sin_enc(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) → (B, delta_dim)."""
        D   = self.delta_dim
        t   = t.float().unsqueeze(-1).clamp(0, 1e5)
        div = torch.exp(
            torch.arange(0, D, 2, device=t.device).float()
            * -(math.log(10000.0) / D)
        )
        enc = torch.zeros(t.size(0), D, device=t.device, dtype=t.dtype)
        enc[:, 0::2] = torch.sin(t * div)
        enc[:, 1::2] = torch.cos(t * div)
        return enc

    def forward(self, relations: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        r    = self.rel_emb(relations)                            # (B, D)
        dt_e = self.delta_proj(self._sin_enc(times))             # (B, D)
        return self.norm(r + dt_e)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  GRU History Encoder
# ══════════════════════════════════════════════════════════════════════════════

class GRUHistoryEncoder(nn.Module):
    """
    Tarihiy faktlarni GRU bilan encode qiladi.

    Kirish: sorted (most-recent-first) [(rel_i, Δt_i)] ketma-ketligi
    Chiqish: gated history signal

    GRU Transformer dan ~5-10x tezroq, sequential data uchun yetarli.
    """

    def __init__(
        self,
        num_relations: int,
        relation_dim:  int,
        hidden_dim:    int,
        delta_dim:     int = 32,
        dropout:       float = 0.1,
    ):
        super().__init__()
        self.rel_emb    = nn.Embedding(num_relations, relation_dim, padding_idx=0)
        self.delta_dim  = delta_dim
        self.delta_proj = nn.Linear(delta_dim, relation_dim)

        # GRU input: [rel_emb | delta_enc]
        self.gru = nn.GRU(
            input_size  = relation_dim * 2,
            hidden_size = hidden_dim,
            num_layers  = 1,
            batch_first = True,
            dropout     = 0.0,
        )
        # Gate: entity embedding + history → combined
        self.gate    = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.drop     = nn.Dropout(dropout)

    def _sin_enc(self, dt: torch.Tensor) -> torch.Tensor:
        """dt: (B, H) → (B, H, delta_dim)."""
        D   = self.delta_dim
        dt  = dt.float().unsqueeze(-1).clamp(0, 1e5)
        div = torch.exp(
            torch.arange(0, D, 2, device=dt.device).float()
            * -(math.log(10000.0) / D)
        )
        enc = torch.zeros(*dt.shape[:2], D, device=dt.device, dtype=dt.dtype)
        enc[..., 0::2] = torch.sin(dt * div)
        enc[..., 1::2] = torch.cos(dt * div)
        return enc

    def forward(
        self,
        hist_rels:  torch.Tensor,   # (B, H)
        hist_dt:    torch.Tensor,   # (B, H)  Δt = t_query - t_hist
        hist_mask:  torch.Tensor,   # (B, H)  True = padding
    ) -> torch.Tensor:
        """Returns: (B, hidden_dim)."""
        B, H = hist_rels.shape

        rel_e   = self.rel_emb(hist_rels)                       # (B, H, rel_dim)
        dt_e    = self.delta_proj(self._sin_enc(hist_dt))       # (B, H, rel_dim)
        x       = torch.cat([rel_e, dt_e], dim=-1)              # (B, H, 2*rel_dim)

        # Valid ketma-ketlik uzunliklari
        lengths = (~hist_mask).sum(dim=1).clamp(min=1).cpu()

        # GRU: pack_padded_sequence tezlashtirib o'qiydi
        packed  = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, h_n  = self.gru(packed)                              # h_n: (1, B, hidden)
        h_out   = self.drop(h_n.squeeze(0))                    # (B, hidden)

        # Tarix bo'lmagan holda nolga qaytaramiz
        has_hist = (~hist_mask).any(dim=1).float().unsqueeze(-1)  # (B, 1)
        return self.out_norm(h_out * has_hist)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Path Encoder — mean rel_embs + attention over paths
# ══════════════════════════════════════════════════════════════════════════════

class PathEncoder(nn.Module):
    """
    Oddiy va tez path encoding.

    Har bir yo'l uchun: rel_emb larni mean pooling
    Yo'llar o'rtasida: o'rganiladigan attention

    Transformer ishlatilmaydi — bu 5-10x tezroq va yetarli darajada sifatli.
    """

    def __init__(
        self,
        num_relations: int,
        relation_dim:  int,
        out_dim:       int,
        dropout:       float = 0.1,
    ):
        super().__init__()
        self.rel_emb    = nn.Embedding(num_relations, relation_dim, padding_idx=0)
        # Path attention scorer
        self.path_score = nn.Linear(relation_dim, 1)
        # Proektsiya
        self.proj       = nn.Sequential(
            nn.Linear(relation_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        path_rels:  torch.Tensor,   # (B, P, L)
        path_masks: torch.Tensor,   # (B, P) True = padding path
    ) -> torch.Tensor:
        B, P, L = path_rels.shape

        # Har bir hop uchun relation embedding
        rel_e = self.rel_emb(path_rels)                         # (B, P, L, rel_dim)

        # Valid hop larni topamiz: rel != 0
        hop_valid = (path_rels != 0).float().unsqueeze(-1)      # (B, P, L, 1)

        # Har bir yo'l uchun mean pooling (L bo'yicha)
        n_valid   = hop_valid.sum(2).clamp(min=1)               # (B, P, 1)
        path_embs = (rel_e * hop_valid).sum(2) / n_valid        # (B, P, rel_dim)

        # Yo'llar o'rtasida attention
        attn_logit = self.path_score(path_embs).squeeze(-1)     # (B, P)
        attn_logit = attn_logit.masked_fill(path_masks, -3e4)
        attn_w     = torch.softmax(attn_logit, dim=-1)          # (B, P)

        # Barcha yo'llar yo'q bo'lsa — nol
        has_path   = (~path_masks).any(dim=1).float().unsqueeze(-1)  # (B, 1)
        attn_w     = attn_w * (~path_masks).float()
        denom      = attn_w.sum(1, keepdim=True).clamp(min=1e-8)
        attn_w     = attn_w / denom

        agg = (attn_w.unsqueeze(-1) * path_embs).sum(1)        # (B, rel_dim)
        agg = agg * has_path

        return self.norm(self.proj(agg))                        # (B, out_dim)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  AURORA — asosiy model
# ══════════════════════════════════════════════════════════════════════════════

class CHRONOSModel(nn.Module):
    """
    AURORA: sodda, tez, ishonchli TKG extrapolation modeli.

    Scoring = query · ent_emb.T  (B×E matmul — bitta, tez)
    NO MLP over all entities!
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
        num_heads:       int   = 8,      # unused (backward compat)
        num_layers:      int   = 2,      # unused (backward compat)
        ffn_dim:         int   = 1024,   # unused (backward compat)
        num_patterns:    int   = 128,    # unused (backward compat)
        num_trtm_bins:   int   = 32,     # unused (backward compat)
        num_negative:    int   = 256,
        dropout:         float = 0.1,
        label_smoothing: float = 0.1,
        w_direct:        float = 0.0,    # disabled — no entity-level MLP
        w_pcl:           float = 0.0,    # disabled — not needed
        use_history:     bool  = True,
        max_history:     int   = 64,
    ):
        super().__init__()

        self.num_entities       = num_entities
        self.num_base_relations = num_relations
        self.total_relations    = num_relations * 2   # + inverse
        self.num_times          = max(num_times, 1)
        self.entity_dim         = entity_dim
        self.hidden_dim         = hidden_dim
        self.use_history        = use_history
        self.label_smoothing    = label_smoothing

        # ── 1. DE Entity Embedding ────────────────────────────────────────────
        self.de_emb = DiachronicEmbedding(num_entities, entity_dim)

        # Scoring uchun statik emb (DE.emb.weight) ishlatiladi
        # Scoring: query · de_emb.emb.weight.T  (tez va stable)

        # ── 2. Temporal Relation Encoding ─────────────────────────────────────
        self.tre = TemporalRelationEncoding(
            self.total_relations, relation_dim, delta_dim
        )

        # ── 3. GRU History Encoder ────────────────────────────────────────────
        if use_history:
            self.hist_enc = GRUHistoryEncoder(
                num_relations = self.total_relations,
                relation_dim  = relation_dim,
                hidden_dim    = hidden_dim,
                delta_dim     = delta_dim // 2,
                dropout       = dropout,
            )
            # History → entity_dim
            self.hist_proj = nn.Linear(hidden_dim, entity_dim)
            self.hist_gate = nn.Sequential(
                nn.Linear(entity_dim * 2, entity_dim),
                nn.Sigmoid(),
            )
            self.hist_norm = nn.LayerNorm(entity_dim)

        # ── 4. Path Encoder ───────────────────────────────────────────────────
        self.path_enc = PathEncoder(
            num_relations = self.total_relations,
            relation_dim  = relation_dim,
            out_dim       = entity_dim,
            dropout       = dropout,
        )

        # ── 5. Query MLP: [s_t | r_t | hist | path] → query ──────────────────
        # Input: entity_dim × 4 (s, r, hist, path) → output: entity_dim
        self.query_mlp = nn.Sequential(
            nn.Linear(entity_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, entity_dim),
            nn.LayerNorm(entity_dim),
        )

        # ── 6. Self-adversarial temperature ───────────────────────────────────
        self.rel_temp = nn.Embedding(self.total_relations * 2, 1)
        nn.init.constant_(self.rel_temp.weight, 1.0)

    # ── Yordamchi ─────────────────────────────────────────────────────────────

    def _fix_rel(self, rel: torch.Tensor) -> torch.Tensor:
        """r+INV_OFFSET → r+num_base_relations."""
        inv_mask = rel >= INV_OFFSET
        if inv_mask.any():
            rel = rel.clone()
            rel[inv_mask] = (rel[inv_mask] - INV_OFFSET) + self.num_base_relations
        return rel.clamp(0, self.total_relations - 1)

    def _t_norm(self, times: torch.Tensor) -> torch.Tensor:
        """Vaqtni 0..1 oralig'iga normallashtirish."""
        return times.float() / float(self.num_times)

    def _encode_query(
        self,
        subjects:   torch.Tensor,   # (B,)
        relations:  torch.Tensor,   # (B,)  already fixed
        times:      torch.Tensor,   # (B,)
        history:    Optional[torch.Tensor],   # (B, H, 3)
        hist_mask:  Optional[torch.Tensor],   # (B, H)
        paths:      torch.Tensor,   # (B, P, L, 3)
        path_masks: torch.Tensor,   # (B, P)
    ) -> torch.Tensor:
        """Returns query vector (B, entity_dim)."""
        B = subjects.size(0)
        t_n = self._t_norm(times)

        # 1. Subject DE embedding
        s_t = self.de_emb(subjects, t_n)                        # (B, entity_dim)

        # 2. Relation temporal encoding (Δt = t itself, yoki 0)
        r_t = self.tre(relations, times)                        # (B, relation_dim=entity_dim)

        # 3. History signal
        if self.use_history and history is not None:
            hist_rels = self._fix_rel(history[:, :, 1])        # (B, H)
            hist_dt   = (times.unsqueeze(-1) - history[:, :, 2]).float().clamp(0)
            h_out     = self.hist_enc(hist_rels, hist_dt, hist_mask)  # (B, hidden)
            h_proj    = self.hist_proj(h_out)                  # (B, entity_dim)
            gate      = self.hist_gate(torch.cat([s_t, h_proj], dim=-1))
            hist_sig  = self.hist_norm(gate * h_proj)
        else:
            hist_sig  = s_t.new_zeros(B, self.entity_dim)

        # 4. Path signal
        path_rels = self._fix_rel(paths[:, :, :, 1])           # (B, P, L)
        path_sig  = self.path_enc(path_rels, path_masks)       # (B, entity_dim)

        # 5. Query MLP
        query = self.query_mlp(
            torch.cat([s_t, r_t, hist_sig, path_sig], dim=-1)
        )                                                       # (B, entity_dim)
        return query

    # ── Forward (training) ────────────────────────────────────────────────────

    def forward(
        self,
        subjects:    torch.Tensor,   # (B,)
        relations:   torch.Tensor,   # (B,)
        objects:     torch.Tensor,   # (B,)
        times:       torch.Tensor,   # (B,)
        paths:       torch.Tensor,   # (B, P, L, 3)
        path_masks:  torch.Tensor,   # (B, P)
        neg_objects: torch.Tensor,   # (B, N)
        history:     Optional[torch.Tensor] = None,
        hist_mask:   Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        B = subjects.size(0)
        rel_fixed = self._fix_rel(relations)

        # Query vector
        query = self._encode_query(
            subjects, rel_fixed, times, history, hist_mask, paths, path_masks
        )                                                       # (B, entity_dim)

        # Scoring: query · ent_emb.T — bitta matmul, tez va stable
        all_ent = self.de_emb.emb.weight                       # (E, entity_dim)
        scores  = (query @ all_ent.T).clamp(-10, 10)           # (B, E)

        # ── Link loss ─────────────────────────────────────────────────────────
        pos_score = scores[torch.arange(B), objects]           # (B,)
        neg_score = scores.gather(1, neg_objects) \
                    if neg_objects.size(1) > 0 \
                    else scores.new_zeros(B, 1)                # (B, N)

        sm = self.label_smoothing
        def bce_ls(logit, label):
            t = torch.full_like(logit, label * (1 - sm) + sm * 0.5)
            return F.binary_cross_entropy_with_logits(logit, t)

        link_loss = bce_ls(pos_score, 1.0) + bce_ls(neg_score, 0.0)

        # ── Self-adversarial loss ─────────────────────────────────────────────
        with torch.no_grad():
            temp  = self.rel_temp(rel_fixed).squeeze(-1).abs().clamp(0.5, 10)
            adv_w = torch.softmax(
                neg_score.detach() * temp.unsqueeze(-1), dim=1
            )

        if neg_objects.size(1) > 0:
            sa_loss = -(adv_w * F.logsigmoid(-neg_score)).sum(1).mean()
        else:
            sa_loss = scores.new_tensor(0.0)

        losses = {
            "link":      link_loss,
            "self_adv":  sa_loss,
            "pcl":       scores.new_tensor(0.0),    # compat
            "ortho_reg": scores.new_tensor(0.0),    # compat
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
        query     = self._encode_query(
            subjects, rel_fixed, times, history, hist_mask, paths, path_masks
        )
        all_ent   = self.de_emb.emb.weight
        return (query @ all_ent.T).clamp(-10, 10)
