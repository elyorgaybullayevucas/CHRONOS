# models/chronos_model.py
"""
CHRONOS: Cross-scale Historical Reasoning and Ontological Network for
         Ordered Sequence-based link prediction.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YANGI KOMPONENTLAR:
  1. TFE  — Temporal Frequency Encoding
             Har bir relatsiya uchun vaqt chastotasi xususiyatlari.
             sin/cos asosida Δt → chastota xususiyatlari.

  2. QATS — Query-Aware Temporal Selection
             Tarix elementlarini cross-attention bilan tanlash.
             Query (s, r, t) → tarix bo'yicha diqqat → tanlangan kontekst.

  3. TRTM — Temporal Relation Transition Memory
             Relatsiya o'tish naqshlarini yodda saqlash.
             (r_prev, r_curr, Δt_bin) → o'tish signali.

  4. PCL  — Path Contrastive Learning
             Yo'l ko'rinishlarida InfoNCE yo'qotmasi.
             To'g'ri yo'llar query ga yaqin, noto'g'rilar uzoq bo'lishi.

ASOSIY KOMPONENTLAR (ORION dan yaxshilangan):
  5. TPL  — Temporal Pattern Library (K=128 naqshlar)
  6. RPE  — Relation Profile Encoding
  7. EI-HT— Entity-Independent History Transformer (self-attn)
  8. RTE  — Relative Δt Encoding (log-space sinusoidal)
  9. 3SF  — Three-Signal Fusion (s-emb + path + temporal signal)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.paths import INV_OFFSET


# ══════════════════════════════════════════════════════════════════════════════
# 1.  TFE — Temporal Frequency Encoding
# ══════════════════════════════════════════════════════════════════════════════

class TemporalFrequencyEncoding(nn.Module):
    """
    Har bir relatsiya uchun vaqtga bog'liq chastota xususiyatlarini hisoblab chiqadi.

    Fikr:
      - Ba'zi relatsiyalar kunlik/haftalik/yillik naqshlarga ega
        (ICEWS: siyosiy voqealar; YAGO: tug'ilgan yillar)
      - Har bir relatsiya uchun K ta chastota parametri o'rganiladi
      - Δt × freq → sin/cos → dimension D ga proektsiya

    Input:
      relations : (B,)   — relatsiya indekslari
      delta_t   : (B,)   — so'rovdan oldingi vaqt farqi (t_query - t_last)
    Output:
      (B, dim)  — chastota xususiyatlari
    """

    def __init__(self, num_relations: int, num_freqs: int = 16, dim: int = 32):
        super().__init__()
        self.num_freqs = num_freqs
        self.dim = dim

        # Har bir relatsiya uchun o'rganiladigan chastotalar
        self.rel_log_freq = nn.Embedding(num_relations, num_freqs)
        # Proektsiya: [sin, cos] × num_freqs → dim
        self.proj = nn.Sequential(
            nn.Linear(num_freqs * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        # Initsializatsiya: log-space chastotalar (1 dan 10_000 gacha)
        nn.init.uniform_(self.rel_log_freq.weight, math.log(1.0), math.log(1e4))

    def forward(
        self,
        relations: torch.Tensor,   # (B,)
        delta_t: torch.Tensor,     # (B,)  float
    ) -> torch.Tensor:
        # O'rganiladigan chastotalar
        log_freq = self.rel_log_freq(relations)           # (B, F)
        freqs = torch.exp(log_freq.clamp(-10, 10))        # (B, F)

        dt = delta_t.float().unsqueeze(-1).clamp(0, 1e5)  # (B, 1)
        phases = freqs * dt                                # (B, F)

        feats = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)  # (B, 2F)
        return self.proj(feats)                            # (B, dim)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  QATS — Query-Aware Temporal Selection
# ══════════════════════════════════════════════════════════════════════════════

class QueryAwareTemporalSelection(nn.Module):
    """
    Tarix elementlarini query vektori bilan cross-attention orqali tanlaydi.

    Muammo:
      Oddiy EI-HT (self-attention) tarix ichidagi muhim faktlarni ajrata olmaydi
      chunki u query ga qaramaydi.

    Yechim:
      - query (s, r, t) ni Key sifatida ishlatib, tarix elementlaridan
        most-relevant faktlarni cross-attention bilan ajratib olamiz
      - Gated residual: original query + gated(attention output)

    Input:
      query    : (B, D)   — so'rov vektori
      hist_embs: (B, H, D)— tarix elementlari embeddingi
      hist_mask: (B, H)   — True qayerlarda padding bor
    Output:
      (B, D)   — tanlangan kontekst bilan boyitilgan so'rov
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # Cross-attention: query → history ga qaraydi
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm  = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        # Gating: qancha history signali qo'shilsin?
        self.gate  = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(
        self,
        query: torch.Tensor,        # (B, D)
        hist_embs: torch.Tensor,    # (B, H, D)
        hist_mask: torch.Tensor,    # (B, H)  True=padding
    ) -> torch.Tensor:
        B, D = query.shape

        # Hech bo'lsa bitta valid tarix elementi bormi?
        has_hist = (~hist_mask).any(dim=1)  # (B,)

        q = query.unsqueeze(1)              # (B, 1, D)

        # Cross-attention
        attn_out, _ = self.cross_attn(
            q, hist_embs, hist_embs,
            key_padding_mask=hist_mask,
        )                                   # (B, 1, D)
        attn_out = attn_out.squeeze(1)      # (B, D)

        # Tarix bo'lmagan batchlarda attn_out ni nollayamiz
        attn_out = attn_out * has_hist.float().unsqueeze(-1)

        # Gated residual
        gate = self.gate(torch.cat([query, attn_out], dim=-1))  # (B, D)
        h    = self.norm(query + gate * attn_out)

        # FFN
        out  = self.norm2(h + self.ffn(h))
        return out  # (B, D)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TRTM — Temporal Relation Transition Memory
# ══════════════════════════════════════════════════════════════════════════════

class TemporalRelationTransitionMemory(nn.Module):
    """
    Relatsiya o'tish naqshlarini Δt bilan birga yodda saqlaydi.

    Fikr:
      "Agar A mamlakatida siyosiy kelishuv bo'lsa (r1), keyin harbiy
       harakatlar (r2) ehtimoli yuqori" — bu naqshni model o'rganishi kerak.
      Vaqt oralig'i (Δt) ham muhim: qisqa vaqtda o'tish vs uzoq muddatda.

    Amalga oshirish:
      - r_src embedding (oldingi relatsiya)
      - r_dst embedding (joriy so'rov relatsiyasi)
      - bin_emb (Δt diskretizatsiyasi, log-space)
      - Trilinear scorer: src ⊗ bin ⊗ dst → o'tish signali

    Input:
      r_curr    : (B,)    — joriy so'rov relatsiyasi
      hist_rels : (B, H)  — tarix relatsiyalari
      hist_dt   : (B, H)  — Δt = t_query - t_prev
      hist_valid: (B, H)  — True qayerlarda haqiqiy element bor
    Output:
      (B, relation_dim)  — o'tish konteksti
    """

    def __init__(
        self,
        num_relations: int,
        relation_dim:  int,
        num_bins:      int = 32,
        dim:           int = 64,
        dropout:       float = 0.1,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.dim      = dim

        # Relatsiya embeddinglari (alohida: asosiy emb bilan aralashmasin)
        self.r_src  = nn.Embedding(num_relations, dim)
        self.r_dst  = nn.Embedding(num_relations, dim)
        # Δt bin embeddinglari (0 = padding/0-delta, 1..num_bins = actual bins)
        self.bin_emb = nn.Embedding(num_bins + 2, dim, padding_idx=0)

        # Trilinear scorer: [src, bin, dst] → transition dim
        self.scorer = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        # Attention (history elementi bo'yicha agregatsiya)
        self.attn_proj = nn.Linear(dim, 1)
        # Output
        self.out_proj  = nn.Linear(dim, relation_dim)
        self.norm      = nn.LayerNorm(relation_dim)

        # Log-space bin chegaralari: 1, 2, 4, ..., 2^(num_bins-1)
        boundaries = torch.logspace(0.0, math.log10(1e4), num_bins - 1)
        self.register_buffer("bin_boundaries", boundaries)

    def _to_bin(self, delta_t: torch.Tensor) -> torch.Tensor:
        """Δt → bin indeksi (1-indexed, 0 = padding)."""
        bins = torch.bucketize(delta_t.float().clamp(0), self.bin_boundaries)
        return (bins + 1).clamp(1, self.num_bins)   # 1..num_bins

    def forward(
        self,
        r_curr:     torch.Tensor,   # (B,)
        hist_rels:  torch.Tensor,   # (B, H)
        hist_dt:    torch.Tensor,   # (B, H)
        hist_valid: torch.Tensor,   # (B, H) bool
    ) -> torch.Tensor:
        B, H = hist_rels.shape

        r_dst_emb = self.r_dst(r_curr)             # (B, D)
        r_src_emb = self.r_src(hist_rels)          # (B, H, D)

        bin_idx   = self._to_bin(hist_dt)          # (B, H)
        bin_emb   = self.bin_emb(bin_idx)          # (B, H, D)

        r_dst_exp = r_dst_emb.unsqueeze(1).expand(-1, H, -1)  # (B, H, D)

        # Trilinear: [r_src, bin, r_dst] → transition signal
        concat    = torch.cat([r_src_emb, bin_emb, r_dst_exp], dim=-1)  # (B, H, 3D)
        trans     = self.scorer(concat)             # (B, H, D)

        # Attention score
        attn_logit = self.attn_proj(trans).squeeze(-1)          # (B, H)
        attn_logit = attn_logit.masked_fill(~hist_valid, -3e4)  # FP16 safe (-1e9 overflow)
        # All-invalid rows → uniform weight (will be zeroed via context)
        has_valid  = hist_valid.any(dim=-1)                     # (B,)
        attn_w     = torch.softmax(attn_logit, dim=-1)          # (B, H)
        attn_w     = attn_w * hist_valid.float()                # zero-out padding

        # Normalize where valid
        denom = attn_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        attn_w = attn_w / denom

        context    = (attn_w.unsqueeze(-1) * trans).sum(dim=1) # (B, D)
        context    = context * has_valid.float().unsqueeze(-1)  # no-hist → 0

        out        = self.out_proj(context)                     # (B, rel_dim)
        return self.norm(out)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PCL — Path Contrastive Learning
# ══════════════════════════════════════════════════════════════════════════════

class PathContrastiveLoss(nn.Module):
    """
    Yo'l ko'rinishlarida InfoNCE yo'qotmasi.

    Fikr:
      - Har bir query (s, r, t) uchun yo'llar bir "pozitiv ko'rinish" beradi.
      - Boshqa querylarning yo'llari "negatif ko'rinish" hisoblanadi.
      - InfoNCE: query embedding o'z yo'l embedding'iga yaqin bo'lsin.

    Bu training signalini kuchaytiradi va yo'l encoderni regularizatsiya qiladi.

    Input:
      query_emb: (B, D) — query representatsiyasi
      path_emb : (B, D) — agregatsiya qilingan yo'l representatsiyasi
    Output:
      scalar loss
    """

    def __init__(self, dim: int, temperature: float = 0.07, proj_dim: int = 128):
        super().__init__()
        self.temperature = temperature

        # Proektsiya boshi (projection head) — contrastive uchun alohida
        self.q_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, proj_dim),
        )
        self.p_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, proj_dim),
        )

    def forward(
        self,
        query_emb: torch.Tensor,    # (B, D)
        path_emb:  torch.Tensor,    # (B, D)
    ) -> torch.Tensor:
        B = query_emb.size(0)
        if B < 2:
            return query_emb.new_tensor(0.0)

        # L2 normallashtirish
        q = F.normalize(self.q_proj(query_emb), dim=-1)   # (B, proj_dim)
        p = F.normalize(self.p_proj(path_emb),  dim=-1)   # (B, proj_dim)

        # O'xshashlik matritsasi
        sim = q @ p.T / self.temperature                   # (B, B)

        # Diagonal — pozitiv juftlar
        labels = torch.arange(B, device=q.device)
        loss_q2p = F.cross_entropy(sim,   labels)
        loss_p2q = F.cross_entropy(sim.T, labels)
        return (loss_q2p + loss_p2q) * 0.5


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TPL — Temporal Pattern Library
# ══════════════════════════════════════════════════════════════════════════════

class TemporalPatternLibrary(nn.Module):
    """
    K ta o'rganiladigan vaqtiy naqshlar to'plami.

    Har bir yo'l representatsiyasi ushbu naqshlar bilan attention orqali
    boyitiladi → yo'lni "qaysi naqsh" ekanini aniqlaydi.
    """

    def __init__(self, num_patterns: int, dim: int, dropout: float = 0.1):
        super().__init__()
        self.patterns = nn.Parameter(torch.randn(num_patterns, dim) * 0.02)
        self.attn_q   = nn.Linear(dim, dim)
        self.attn_k   = nn.Linear(dim, dim)
        self.attn_v   = nn.Linear(dim, dim)
        self.scale    = dim ** -0.5
        self.out_proj = nn.Linear(dim, dim)
        self.norm     = nn.LayerNorm(dim)
        self.drop     = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D) — yo'l representatsiyasi."""
        K, D = self.patterns.shape
        q = self.attn_q(x).unsqueeze(1)                          # (B, 1, D)
        k = self.attn_k(self.patterns).unsqueeze(0)              # (1, K, D)
        v = self.attn_v(self.patterns).unsqueeze(0)              # (1, K, D)
        score = (q * k).sum(-1) * self.scale                     # (B, K)
        w     = torch.softmax(score, dim=-1)                     # (B, K)
        agg   = (w.unsqueeze(-1) * v).sum(dim=1)                 # (B, D)
        return self.norm(x + self.drop(self.out_proj(agg)))


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Path Encoder — EI-HT (Entity-Independent History Transformer)
# ══════════════════════════════════════════════════════════════════════════════

class PathEncoder(nn.Module):
    """
    Har bir yo'lni (sequence of hops) Transformer bilan encode qiladi.

    Entity-Independent: node embeddinglardan mustaqil (faqat relation + Δt).
    Bu generalizatsiyani oshiradi va yangi entitylarga ko'chirish imkonini beradi.

    Bir hop = (neighbor_entity, relation, time):
      - relation embedding
      - Δt sinusoidal encoding
      - entity embedding (QO'SHILMAYDI — entity-independent)
    """

    def __init__(
        self,
        num_relations: int,
        relation_dim:  int,
        hidden_dim:    int,
        num_heads:     int = 4,
        num_layers:    int = 2,
        delta_dim:     int = 64,
        max_len:       int = 3,
        dropout:       float = 0.1,
    ):
        super().__init__()
        self.rel_emb    = nn.Embedding(num_relations, relation_dim)
        self.delta_proj = nn.Linear(delta_dim, relation_dim)

        # Δt log-sinusoidal
        self.delta_dim  = delta_dim

        input_dim = relation_dim * 2   # rel + delta
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool        = nn.Linear(hidden_dim, hidden_dim)
        self.norm        = nn.LayerNorm(hidden_dim)

    def _delta_encoding(self, dt: torch.Tensor) -> torch.Tensor:
        """dt: (B, L) → (B, L, delta_dim)."""
        D = self.delta_dim
        dt = dt.float().unsqueeze(-1).clamp(0, 1e5)             # (B, L, 1)
        div = torch.exp(
            torch.arange(0, D, 2, device=dt.device).float()
            * -(math.log(10000.0) / D)
        )                                                        # (D//2,)
        args = dt * div.view(1, 1, -1)                           # (B, L, D//2)
        enc  = torch.zeros(*dt.shape[:2], D, device=dt.device)
        enc[..., 0::2] = torch.sin(args)
        enc[..., 1::2] = torch.cos(args)
        return enc

    def forward(
        self,
        path_rels:  torch.Tensor,   # (B, num_paths, max_len)
        path_times: torch.Tensor,   # (B, num_paths, max_len)
        t_query:    torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """Returns: (B, num_paths, hidden_dim)."""
        B, P, L = path_rels.shape

        # Flatten paths
        rel_flat   = path_rels.view(B * P, L)                   # (BP, L)
        time_flat  = path_times.view(B * P, L)                  # (BP, L)
        tq_flat    = t_query.unsqueeze(1).expand(-1, P).reshape(B * P)  # (BP,)

        # Δt = t_query - t_edge (vaqtiy farq)
        dt_flat = (tq_flat.unsqueeze(-1) - time_flat).clamp(0)  # (BP, L)

        # Relation embedding
        rel_e  = self.rel_emb(rel_flat)                         # (BP, L, rel_dim)
        # Delta encoding
        delta_e = self._delta_encoding(dt_flat)                 # (BP, L, delta_dim)
        delta_e = self.delta_proj(delta_e)                      # (BP, L, rel_dim)

        # Concatenate
        x = torch.cat([rel_e, delta_e], dim=-1)                 # (BP, L, rel_dim*2)
        x = self.input_proj(x)                                   # (BP, L, hidden_dim)

        # Padding maskasi: rel==0 AND time==0 → padding
        pad_mask = (rel_flat == 0) & (time_flat == 0)           # (BP, L)

        # Transformer
        h = self.transformer(x, src_key_padding_mask=pad_mask)  # (BP, L, hidden_dim)

        # Global pooling (mean over valid tokens)
        valid_f  = (~pad_mask).float().unsqueeze(-1)            # (BP, L, 1)
        denom    = valid_f.sum(1).clamp(min=1)                  # (BP, 1)
        pooled   = (h * valid_f).sum(1) / denom                 # (BP, hidden_dim)
        pooled   = self.norm(self.pool(pooled))                  # (BP, hidden_dim)

        return pooled.view(B, P, -1)                            # (B, P, hidden_dim)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  History Encoder (EI-HT asosi)
# ══════════════════════════════════════════════════════════════════════════════

class HistoryEncoder(nn.Module):
    """
    Entity s ning vaqtiy tarixini encode qiluvchi self-attention modul.

    Bu QATS dan OLDIN keladi:
      raw history → self-attention → (B, H, D)
    Keyin QATS bu encoded tarixdan query ga mos elementlarni tanlaydi.
    """

    def __init__(
        self,
        num_relations: int,
        relation_dim:  int,
        hidden_dim:    int,
        num_heads:     int = 4,
        num_layers:    int = 2,
        delta_dim:     int = 64,
        dropout:       float = 0.1,
    ):
        super().__init__()
        self.rel_emb    = nn.Embedding(num_relations, relation_dim, padding_idx=0)
        self.delta_dim  = delta_dim
        self.delta_proj = nn.Linear(delta_dim, relation_dim)

        # Input: rel + delta → hidden
        self.input_proj = nn.Linear(relation_dim * 2, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(hidden_dim)

        # History → entity embedding space ga o'tkazish
        self.to_entity   = nn.Linear(hidden_dim, hidden_dim)

    def _delta_encoding(self, dt: torch.Tensor) -> torch.Tensor:
        """dt: (B, H) → (B, H, delta_dim)."""
        D = self.delta_dim
        dt  = dt.float().unsqueeze(-1).clamp(0, 1e5)            # (B, H, 1)
        div = torch.exp(
            torch.arange(0, D, 2, device=dt.device).float()
            * -(math.log(10000.0) / D)
        )
        args = dt * div.view(1, 1, -1)
        enc  = torch.zeros(*dt.shape[:2], D, device=dt.device)
        enc[..., 0::2] = torch.sin(args)
        enc[..., 1::2] = torch.cos(args)
        return enc

    def forward(
        self,
        hist_rels:  torch.Tensor,   # (B, H)
        hist_dt:    torch.Tensor,   # (B, H)  Δt = t_query - t_hist
        hist_mask:  torch.Tensor,   # (B, H)  True = padding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          hist_embs : (B, H, hidden_dim) — encoded tarix elementlari (QATS uchun)
          hist_pooled: (B, hidden_dim)   — agregatsiya qilingan signal (3SF uchun)
        """
        rel_e   = self.rel_emb(hist_rels)                       # (B, H, rel_dim)
        delta_e = self._delta_encoding(hist_dt)                  # (B, H, delta_dim)
        delta_e = self.delta_proj(delta_e)                       # (B, H, rel_dim)

        x   = self.input_proj(torch.cat([rel_e, delta_e], dim=-1))  # (B, H, hidden_dim)
        h   = self.transformer(x, src_key_padding_mask=hist_mask)    # (B, H, hidden_dim)
        h   = self.norm(h)

        # Agregatsiya (mean over valid)
        valid   = (~hist_mask).float().unsqueeze(-1)             # (B, H, 1)
        denom   = valid.sum(1).clamp(min=1)
        pooled  = (h * valid).sum(1) / denom                    # (B, hidden_dim)
        pooled  = self.to_entity(pooled)

        return h, pooled


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Relation Profile Encoding (RPE)
# ══════════════════════════════════════════════════════════════════════════════

class RelationProfileEncoding(nn.Module):
    """
    Relatsiya uchun vaqtiy profil: r × Δt → boyitilgan relatsiya ko'rinishi.

    Har bir relatsiya uchun vaqt bo'yicha "xatti-harakat profili" o'rganiladi.
    Bu relatsiya embeddingini vaqtiy kontekst bilan boyitadi.
    """

    def __init__(
        self,
        num_relations: int,
        relation_dim:  int,
        delta_dim:     int = 64,
        dropout:       float = 0.1,
    ):
        super().__init__()
        self.base_emb    = nn.Embedding(num_relations, relation_dim)
        self.delta_dim   = delta_dim
        self.delta_proj  = nn.Linear(delta_dim, relation_dim)
        self.fusion      = nn.Sequential(
            nn.Linear(relation_dim * 2, relation_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(relation_dim, relation_dim),
        )
        self.norm = nn.LayerNorm(relation_dim)

    def _delta_enc(self, scalar_dt: torch.Tensor) -> torch.Tensor:
        """scalar_dt: (B,) → (B, delta_dim)."""
        D   = self.delta_dim
        dt  = scalar_dt.float().unsqueeze(-1).clamp(0, 1e5)     # (B, 1)
        div = torch.exp(
            torch.arange(0, D, 2, device=dt.device).float()
            * -(math.log(10000.0) / D)
        )
        enc = torch.zeros(dt.size(0), D, device=dt.device)
        enc[:, 0::2] = torch.sin(dt * div)
        enc[:, 1::2] = torch.cos(dt * div)
        return enc

    def forward(
        self,
        relations: torch.Tensor,   # (B,)
        times:     torch.Tensor,   # (B,)   — mutlaq vaqt indeksi
        max_t:     int,
    ) -> torch.Tensor:
        rel_e  = self.base_emb(relations)                        # (B, rel_dim)
        # Δt = max_t - t (vaqt oralig'i)
        dt     = (max_t - times).float().clamp(0)
        delta  = self.delta_proj(self._delta_enc(dt))            # (B, rel_dim)
        return self.norm(self.fusion(torch.cat([rel_e, delta], dim=-1)))


# ══════════════════════════════════════════════════════════════════════════════
# 9.  Scoring Head
# ══════════════════════════════════════════════════════════════════════════════

class ScoringHead(nn.Module):
    """
    Yakuniy link prediction scoring.

    Score(s, r, o, t) = query_vec · entity_emb[o]

    query_vec  = 3SF(entity_signal, path_signal, temporal_signal)
    """

    def __init__(self, hidden_dim: int, entity_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, entity_dim),
        )
        self.norm = nn.LayerNorm(entity_dim)

    def forward(self, query_vec: torch.Tensor) -> torch.Tensor:
        """query_vec: (B, hidden_dim) → (B, entity_dim)."""
        return self.norm(self.proj(query_vec))


# ══════════════════════════════════════════════════════════════════════════════
# 10. CHRONOS — Asosiy model
# ══════════════════════════════════════════════════════════════════════════════

class CHRONOSModel(nn.Module):
    """
    CHRONOS: Cross-scale Historical Reasoning and Ontological Network for
             Ordered Sequence-based link prediction.

    Arxitektura:
      INPUT  : (s, r, t, paths, history)
               ↓
      EMBED  : ent_emb[s] + RPE(r, t) + TFE(r, Δt_tfe)
               ↓
      PATHS  : PathEncoder → TPL → path_signal (B, hidden)
               ↓
      HISTORY: HistoryEncoder → QATS(query) → TRTM → history_signal
               ↓
      FUSION : 3SF(entity, path, history) → query_vec
               ↓
      SCORE  : query_vec · ent_emb[ALL] → (B, num_entities)
               ↓
      LOSSES : LinkPred + SelfAdv + PCL + OrthoReg
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
        tfe_dim:         int   = 32,
        trtm_dim:        int   = 64,
        num_heads:       int   = 8,
        num_layers:      int   = 2,
        ffn_dim:         int   = 1024,
        num_patterns:    int   = 128,
        num_trtm_bins:   int   = 32,
        num_negative:    int   = 256,
        dropout:         float = 0.1,
        label_smoothing: float = 0.1,
        w_direct:        float = 1.0,
        w_pcl:           float = 0.1,
        use_history:     bool  = True,
        max_history:     int   = 64,
    ):
        super().__init__()

        self.num_entities      = num_entities
        self.num_base_relations = num_relations
        # Reciprocal: jami relatsiyalar = original + inverse
        total_relations        = num_relations * 2
        self.total_relations   = total_relations
        self.num_times         = num_times
        self.hidden_dim        = hidden_dim
        self.entity_dim        = entity_dim
        self.use_history       = use_history
        self.max_history       = max_history
        self.w_direct          = w_direct
        self.w_pcl             = w_pcl
        self.label_smoothing   = label_smoothing

        # ── Entity embeddinglari ──────────────────────────────────────────────
        self.ent_emb = nn.Embedding(num_entities, entity_dim)
        nn.init.xavier_uniform_(self.ent_emb.weight)

        # ── Relation embeddinglari (asosiy + inverse) ─────────────────────────
        self.rel_emb = nn.Embedding(total_relations, relation_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        # ── RPE — Relation Profile Encoding ───────────────────────────────────
        self.rpe = RelationProfileEncoding(
            total_relations, relation_dim, delta_dim, dropout
        )

        # ── TFE — Temporal Frequency Encoding ─────────────────────────────────
        self.tfe = TemporalFrequencyEncoding(
            total_relations, num_freqs=16, dim=tfe_dim
        )

        # ── Path Encoder + TPL ────────────────────────────────────────────────
        self.path_encoder = PathEncoder(
            num_relations=total_relations,
            relation_dim=relation_dim,
            hidden_dim=hidden_dim,
            num_heads=min(num_heads, 4),
            num_layers=min(num_layers, 2),
            delta_dim=delta_dim,
            dropout=dropout,
        )
        self.tpl = TemporalPatternLibrary(num_patterns, hidden_dim, dropout)

        # Path aggregation (mean + max → hidden)
        self.path_agg = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # ── History Encoder + QATS + TRTM ────────────────────────────────────
        if use_history:
            self.hist_encoder = HistoryEncoder(
                num_relations=total_relations,
                relation_dim=relation_dim,
                hidden_dim=hidden_dim,
                num_heads=min(num_heads, 4),
                num_layers=min(num_layers, 2),
                delta_dim=delta_dim,
                dropout=dropout,
            )
            self.qats = QueryAwareTemporalSelection(
                dim=hidden_dim,
                num_heads=min(num_heads, 4),
                dropout=dropout,
            )
            self.trtm = TemporalRelationTransitionMemory(
                num_relations=total_relations,
                relation_dim=relation_dim,
                num_bins=num_trtm_bins,
                dim=trtm_dim,
                dropout=dropout,
            )
            # TRTM output → hidden_dim
            self.trtm_proj = nn.Linear(relation_dim, hidden_dim)

            # history_signal = QATS + TRTM → gate → hidden
            self.hist_gate  = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
            self.hist_norm  = nn.LayerNorm(hidden_dim)

        # ── 3SF — Three-Signal Fusion ─────────────────────────────────────────
        # Signal 1: entity embedding → hidden
        self.ent_proj = nn.Linear(entity_dim + relation_dim + tfe_dim, hidden_dim)

        # Signal fusion: entity + path + history → hidden
        # MUHIM: har doim 3 signal ishlatiladi; use_history=False holda hist_signal=0
        fusion_in = hidden_dim * 3
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # ── Scoring Head ──────────────────────────────────────────────────────
        self.scorer = ScoringHead(hidden_dim, entity_dim, dropout)

        # Direct scoring (entity_dim → 1 per candidate, optional)
        self.direct_head = nn.Sequential(
            nn.Linear(entity_dim * 2 + relation_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # ── PCL — Path Contrastive Learning ───────────────────────────────────
        self.pcl = PathContrastiveLoss(dim=hidden_dim, temperature=0.07, proj_dim=128)

        # ── Self-adversarial temp ─────────────────────────────────────────────
        self.rel_temp = nn.Embedding(total_relations * 2, 1)
        nn.init.constant_(self.rel_temp.weight, 1.0)

        # ── TPL ortho regularization ──────────────────────────────────────────
        # (tpl.patterns: K × D)

    # ── Yordamchi: relation indeksini to'g'rilash ─────────────────────────────

    def _fix_rel(self, rel: torch.Tensor) -> torch.Tensor:
        """
        build_graph da inverse edges r+INV_OFFSET bilan belgilanadi.
        Buni r+num_base_relations ga o'zgartiramiz va clamp qilamiz.
        """
        inv_mask = rel >= INV_OFFSET
        if inv_mask.any():
            rel = rel.clone()
            rel[inv_mask] = (rel[inv_mask] - INV_OFFSET) + self.num_base_relations
        return rel.clamp(0, self.total_relations - 1)

    # ── Path encoding ─────────────────────────────────────────────────────────

    def _encode_paths(
        self,
        paths:      torch.Tensor,   # (B, P, L, 3)  [node, rel, time]
        path_masks: torch.Tensor,   # (B, P)         True=padding path
        t_query:    torch.Tensor,   # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          path_signal: (B, hidden_dim)
          path_pooled: (B, hidden_dim)  — PCL uchun
        """
        B, P, L, _ = paths.shape

        path_rels  = self._fix_rel(paths[:, :, :, 1])           # (B, P, L)
        path_times = paths[:, :, :, 2]                          # (B, P, L)

        # Encode: (B, P, hidden_dim)
        embs = self.path_encoder(path_rels, path_times, t_query)

        # TPL: har bir yo'lni naqshlar bilan boyitamiz
        B_, P_, D_ = embs.shape
        embs_flat  = embs.view(B_ * P_, D_)
        embs_flat  = self.tpl(embs_flat)
        embs       = embs_flat.view(B_, P_, D_)

        # Maskelangan yo'llarni nollash
        valid_paths = (~path_masks).float().unsqueeze(-1)        # (B, P, 1)
        embs        = embs * valid_paths

        # Agregatsiya: mean + max
        n_valid  = valid_paths.sum(1).clamp(min=1)               # (B, 1)
        mean_p   = embs.sum(1) / n_valid                         # (B, D)
        # max (padding → -inf masking)
        masked   = embs - (1 - valid_paths) * 3e4  # FP16 safe
        max_p    = masked.max(1).values                          # (B, D)

        path_signal = self.path_agg(torch.cat([mean_p, max_p], dim=-1))  # (B, D)
        return path_signal, mean_p   # mean_p → PCL uchun

    # ── History processing ────────────────────────────────────────────────────

    def _process_history(
        self,
        history:   torch.Tensor,   # (B, H, 3)  [nb, rel, time]
        hist_mask: torch.Tensor,   # (B, H)     True=padding
        t_query:   torch.Tensor,   # (B,)
        r_query:   torch.Tensor,   # (B,)       so'rov relatsiyasi
        query_vec: torch.Tensor,   # (B, D)     so'rov vektori (QATS uchun)
    ) -> torch.Tensor:
        """Returns: history_signal (B, hidden_dim)."""
        if not self.use_history or history is None:
            return query_vec.new_zeros(query_vec.size(0), self.hidden_dim)

        hist_rels = self._fix_rel(history[:, :, 1])              # (B, H)
        hist_times = history[:, :, 2]                            # (B, H)

        # Δt = t_query - t_hist
        dt = (t_query.unsqueeze(-1) - hist_times).float().clamp(0)  # (B, H)

        # 1. HistoryEncoder: raw → encoded (B, H, D)
        hist_embs, hist_pooled = self.hist_encoder(hist_rels, dt, hist_mask)

        # 2. QATS: query × history → query-aware history
        qats_out = self.qats(query_vec, hist_embs, hist_mask)    # (B, D)

        # 3. TRTM: relatsiya o'tish signali
        trtm_out = self.trtm(r_query, hist_rels, dt, ~hist_mask)  # (B, rel_dim)
        trtm_h   = self.trtm_proj(trtm_out)                      # (B, D)

        # 4. QATS + TRTM → gate → history_signal
        gate     = self.hist_gate(torch.cat([qats_out, trtm_h], dim=-1))  # (B, D)
        combined = gate * qats_out + (1 - gate) * trtm_h
        return self.hist_norm(combined)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        subjects:    torch.Tensor,   # (B,)
        relations:   torch.Tensor,   # (B,)
        objects:     torch.Tensor,   # (B,)
        times:       torch.Tensor,   # (B,)
        paths:       torch.Tensor,   # (B, P, L, 3)
        path_masks:  torch.Tensor,   # (B, P)
        neg_objects: torch.Tensor,   # (B, N)
        history:     Optional[torch.Tensor] = None,   # (B, H, 3)
        hist_mask:   Optional[torch.Tensor] = None,   # (B, H)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        B = subjects.size(0)
        rel_fixed = self._fix_rel(relations)                     # (B,)

        # ── 1. Embedding signali ──────────────────────────────────────────────
        s_emb   = self.ent_emb(subjects)                         # (B, ent_dim)
        r_emb   = self.rpe(rel_fixed, times, self.num_times)     # (B, rel_dim)

        # TFE: so'rov vaqtini t_max dan Δt sifatida ishlatamiz
        tfe_feat = self.tfe(rel_fixed, (self.num_times - times).float())  # (B, tfe_dim)

        ent_signal = self.ent_proj(torch.cat([s_emb, r_emb, tfe_feat], dim=-1))  # (B, D)

        # ── 2. Path signali ───────────────────────────────────────────────────
        path_signal, path_for_pcl = self._encode_paths(paths, path_masks, times)

        # ── 3. Asosiy query vektori (history dan oldin, hist_signal=0)  ─────────
        # Har doim 3 signal: ent + path + zeros  (fusion_in = hidden*3)
        pre_query = self.fusion(torch.cat(
            [ent_signal, path_signal, torch.zeros_like(ent_signal)], dim=-1
        ))

        # ── 4. History signali (QATS + TRTM) ─────────────────────────────────
        if self.use_history and history is not None:
            hist_signal = self._process_history(
                history, hist_mask, times, rel_fixed, pre_query
            )
        else:
            hist_signal = ent_signal.new_zeros(B, self.hidden_dim)

        # ── 5. 3SF: Three-Signal Fusion ───────────────────────────────────────
        query_vec = self.fusion(torch.cat([ent_signal, path_signal, hist_signal], dim=-1))

        # ── 6. Scoring ────────────────────────────────────────────────────────
        query_proj = self.scorer(query_vec)                      # (B, ent_dim)

        # Barcha entity larga o'xshashlik
        all_ent  = self.ent_emb.weight                          # (E, ent_dim)
        scores   = query_proj @ all_ent.T                       # (B, E)

        # Direct scoring (ixtiyoriy qo'shimcha signal)
        if self.w_direct > 0:
            o_emb   = all_ent                                   # (E, ent_dim)
            r_exp   = r_emb.unsqueeze(1).expand(-1, self.num_entities, -1)
            q_exp   = query_proj.unsqueeze(1).expand(-1, self.num_entities, -1)
            o_exp   = o_emb.unsqueeze(0).expand(B, -1, -1)
            direct  = self.direct_head(
                torch.cat([q_exp, o_exp, r_exp], dim=-1)
            ).squeeze(-1)                                        # (B, E)
            scores  = scores + self.w_direct * direct

        # FP16 overflow oldini olish: scorlarni clamp qilamiz
        scores = scores.clamp(-20, 20)

        # ── 7. Link Prediction Loss (NSS + label smoothing) ──────────────────
        # Pozitiv ob'ekt + N ta negatif
        pos_score = scores[torch.arange(B), objects]            # (B,)

        if neg_objects.size(1) > 0:
            neg_score = scores.gather(1, neg_objects)           # (B, N)
        else:
            neg_score = torch.zeros(B, 1, device=scores.device)

        # Label smoothing bilan BCE yo'qotmasi
        def bce_ls(logit: torch.Tensor, label: float) -> torch.Tensor:
            smooth = self.label_smoothing
            t      = torch.full_like(logit, label * (1 - smooth) + smooth * 0.5)
            return F.binary_cross_entropy_with_logits(logit, t)

        link_loss = bce_ls(pos_score, 1.0) + bce_ls(neg_score, 0.0)

        # ── 8. Self-adversarial Loss ──────────────────────────────────────────
        # Negatiflarga og'irlik: softmax(neg_score * temp)
        with torch.no_grad():
            temp  = self.rel_temp(rel_fixed).squeeze(-1).abs().clamp(0.5, 10)
            adv_w = torch.softmax(neg_score.detach() * temp.unsqueeze(-1), dim=1)

        sa_loss = -(adv_w * F.logsigmoid(-neg_score)).sum(1).mean() \
                  if neg_objects.size(1) > 0 else scores.new_tensor(0.0)

        # ── 9. PCL — Path Contrastive Loss ────────────────────────────────────
        pcl_loss = self.pcl(query_vec.detach(), path_for_pcl)

        # ── 10. Ortho Regularization (TPL patterns) ────────────────────────────
        n_base  = min(self.num_base_relations, 64)
        pts     = F.normalize(self.tpl.patterns[:n_base], dim=-1)
        gram    = pts @ pts.T
        eye     = torch.eye(n_base, device=pts.device)
        ortho   = F.mse_loss(gram, eye)

        losses = {
            "link":      link_loss,
            "self_adv":  sa_loss,
            "pcl":       pcl_loss,
            "ortho_reg": ortho,
        }
        return scores, losses

    # ── Predict (evaluation) ──────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        subjects:   torch.Tensor,   # (B,)
        relations:  torch.Tensor,   # (B,)
        times:      torch.Tensor,   # (B,)
        paths:      torch.Tensor,   # (B, P, L, 3)
        path_masks: torch.Tensor,   # (B, P)
        history:    Optional[torch.Tensor] = None,
        hist_mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns: scores (B, num_entities)."""
        B = subjects.size(0)
        rel_fixed = self._fix_rel(relations)

        s_emb    = self.ent_emb(subjects)
        r_emb    = self.rpe(rel_fixed, times, self.num_times)
        tfe_feat = self.tfe(rel_fixed, (self.num_times - times).float())

        ent_signal  = self.ent_proj(torch.cat([s_emb, r_emb, tfe_feat], dim=-1))
        path_signal, _ = self._encode_paths(paths, path_masks, times)

        # Pre-query (history dan oldin)
        pre_query = self.fusion(torch.cat(
            [ent_signal, path_signal, torch.zeros_like(ent_signal)], dim=-1
        ))

        if self.use_history and history is not None:
            hist_signal = self._process_history(
                history, hist_mask, times, rel_fixed, pre_query
            )
        else:
            hist_signal = ent_signal.new_zeros(B, self.hidden_dim)

        query_vec  = self.fusion(torch.cat([ent_signal, path_signal, hist_signal], dim=-1))
        query_proj = self.scorer(query_vec)

        all_ent = self.ent_emb.weight
        scores  = query_proj @ all_ent.T

        if self.w_direct > 0:
            r_exp  = r_emb.unsqueeze(1).expand(-1, self.num_entities, -1)
            q_exp  = query_proj.unsqueeze(1).expand(-1, self.num_entities, -1)
            o_exp  = all_ent.unsqueeze(0).expand(B, -1, -1)
            direct = self.direct_head(
                torch.cat([q_exp, o_exp, r_exp], dim=-1)
            ).squeeze(-1)
            scores = scores + self.w_direct * direct

        scores = scores.clamp(-20, 20)

        return scores
