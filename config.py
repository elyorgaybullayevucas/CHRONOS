# config.py
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset:       str = "ICEWS18"
    data_dir:      str = "data"
    num_entities:  int = 0
    num_relations: int = 0
    num_times:     int = 0

    # ── Model ─────────────────────────────────────────────────────────────────
    entity_dim:   int   = 64     # argparse default bilan mos (64)
    relation_dim: int   = 64
    hidden_dim:   int   = 256
    delta_dim:    int   = 64
    dropout:      float = 0.1

    # ── Path sampling ─────────────────────────────────────────────────────────
    num_paths:    int = 6
    max_path_len: int = 3
    num_negative: int = 256

    # ── History ───────────────────────────────────────────────────────────────
    use_history:    bool = True
    max_history:    int  = 10     # NEXUS snapshot history length (DaeMon: 10-25)
    use_reciprocal: bool = True

    # ── Training ──────────────────────────────────────────────────────────────
    num_epochs:      int   = 30    # argparse default bilan mos (30)
    batch_size:      int   = 32    # argparse default bilan mos (32)
    learning_rate:   float = 5e-4  # argparse default bilan mos (5e-4)
    weight_decay:    float = 1e-5  # argparse default bilan mos (1e-5)
    grad_clip:       float = 1.0
    label_smoothing: float = 0.1

    # ── Loss weights ──────────────────────────────────────────────────────────
    w_link:      float = 1.0
    w_self_adv:  float = 0.5
    w_pcl:       float = 0.0   # disabled
    w_ortho_reg: float = 0.0   # disabled

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

    # backward compat
    num_heads:     int   = 8
    num_layers:    int   = 2
    ffn_dim:       int   = 1024
    num_patterns:  int   = 128
    num_trtm_bins: int   = 32
    tfe_dim:       int   = 32
    trtm_dim:      int   = 64
    w_direct:      float = 0.0
    w_pattern_div: float = 0.0
    w_contrastive: float = 0.0


DATASET_STATS = {
    "ICEWS14": {"num_entities": 7128,  "num_relations": 230,  "num_times": 365},
    "ICEWS18": {"num_entities": 23033, "num_relations": 256,  "num_times": 304},
    "WIKI":    {"num_entities": 12554, "num_relations": 24,   "num_times": 232},
    "YAGO":    {"num_entities": 10623, "num_relations": 10,   "num_times": 189},
    "YAGOs":   {"num_entities": 10623, "num_relations": 10,   "num_times": 189},
    "GDELT":   {"num_entities": 7691,  "num_relations": 240,  "num_times": 2975},
}
