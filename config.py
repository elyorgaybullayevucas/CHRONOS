# config.py
"""
CHRONOS — konfiguratsiya sinfi.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset:      str = "ICEWS18"
    data_dir:     str = "data"
    num_entities:  int = 0
    num_relations: int = 0
    num_times:     int = 0

    # ── Model o'lchamlari ─────────────────────────────────────────────────────
    entity_dim:   int = 256
    relation_dim: int = 256
    hidden_dim:   int = 512
    delta_dim:    int = 64     # Δt encoding dimension
    tfe_dim:      int = 32     # TFE output dimension
    trtm_dim:     int = 64     # TRTM internal dimension
    num_heads:    int = 8
    num_layers:   int = 2
    ffn_dim:      int = 1024
    dropout:      float = 0.1

    # ── TPL (Temporal Pattern Library) ────────────────────────────────────────
    num_patterns: int = 128    # K — pattern soni

    # ── TRTM (Temporal Relation Transition Memory) ────────────────────────────
    num_trtm_bins: int = 32    # Δt diskretizatsiya bin soni

    # ── Path sampling ─────────────────────────────────────────────────────────
    num_paths:    int = 8
    max_path_len: int = 3

    # ── Negative sampling ─────────────────────────────────────────────────────
    num_negative: int = 256

    # ── Tarix (history) ───────────────────────────────────────────────────────
    use_history:  bool = True
    max_history:  int  = 64

    # ── Reciprocal triples ────────────────────────────────────────────────────
    use_reciprocal: bool = True

    # ── O'qitish ──────────────────────────────────────────────────────────────
    num_epochs:      int   = 50
    batch_size:      int   = 512
    learning_rate:   float = 3e-4
    weight_decay:    float = 1e-4
    grad_clip:       float = 1.0
    label_smoothing: float = 0.1

    # ── Yo'qotish og'irliklari ─────────────────────────────────────────────────
    w_link:      float = 1.0
    w_self_adv:  float = 0.5
    w_pcl:       float = 0.1   # Path Contrastive Learning
    w_ortho_reg: float = 0.001

    # ── Runtime ───────────────────────────────────────────────────────────────
    device:      str  = "cuda"
    fp16:        bool = True
    seed:        int  = 42
    num_workers: int  = 8
    save_dir:    str  = "checkpoints"
    log_dir:     str  = "logs"
    resume:      Optional[str] = None
    eval_every:  int  = 1
    filter_flag: bool = True
    hits_at_k:   List[int] = field(default_factory=lambda: [1, 3, 10])


# ── Dataset statistikalari ────────────────────────────────────────────────────
DATASET_STATS = {
    "ICEWS14": {"num_entities": 7128,  "num_relations": 230,  "num_times": 365},
    "ICEWS18": {"num_entities": 23033, "num_relations": 256,  "num_times": 304},
    "WIKI":    {"num_entities": 12554, "num_relations": 24,   "num_times": 232},
    "YAGO":    {"num_entities": 10623, "num_relations": 10,   "num_times": 189},
    "YAGOs":   {"num_entities": 10623, "num_relations": 10,   "num_times": 189},
    "GDELT":   {"num_entities": 7691,  "num_relations": 240,  "num_times": 2975},
}
