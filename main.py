# main.py
"""
NEXUS — Neural EXtrapolation with Unified Snapshot-graphs.

DaeMon dan yaxshiroq TKG extrapolation modeli.
  - DaeMon: BCE + 64 neg, entity-free, d=64
  - NEXUS:  1-N CE, DE entity embeddings, d=128, DaeMon arxitekturasi

Foydalanish:
    python main.py --dataset YAGO
    python main.py --dataset WIKI
    python main.py --dataset ICEWS18
    python main.py --dataset ICEWS14
    python main.py --dataset GDELT
    python main.py --dataset YAGO --entity_dim 64  # kichik model
"""
import argparse
import os
import random

import numpy as np
import torch

from config import Config
from data.dataset import CHRONOSDataset
from data.snapshot import SnapshotGraph
from models.chronos_model import CHRONOSModel
from trainers.trainer import CHRONOSTrainer
from utils.logging import get_logger


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_args() -> Config:
    p = argparse.ArgumentParser("NEXUS — Neural TKG Extrapolation")

    p.add_argument("--dataset",    default="ICEWS18",
                   choices=["ICEWS14", "ICEWS18", "WIKI", "YAGO", "YAGOs", "GDELT"])
    p.add_argument("--data_dir",   default="data")
    p.add_argument("--entity_dim", type=int,   default=64)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--max_history",type=int,   default=10,    dest="max_history")
    p.add_argument("--epochs",     type=int,   default=30,    dest="num_epochs")
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=5e-4,  dest="learning_rate")
    p.add_argument("--weight_decay",type=float,default=1e-5)
    p.add_argument("--grad_clip",  type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--w_link",     type=float, default=1.0)
    p.add_argument("--use_reciprocal", action="store_true")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--save_dir",   default="checkpoints")
    p.add_argument("--log_dir",    default="logs")
    p.add_argument("--resume",     default=None)
    p.add_argument("--no_fp16",    action="store_true")
    p.add_argument("--eval_every", type=int,   default=1)

    args = p.parse_args()
    cfg  = Config()
    for k, v in vars(args).items():
        if k == "no_fp16":
            cfg.fp16 = not v
        elif k == "use_reciprocal":
            if v:
                cfg.use_reciprocal = True
        elif hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _preset(cfg: Config, logger) -> None:
    """
    Dataset-specific hyperparameters.
    DaeMon paper settings + NEXUS tuning.
    """
    if cfg.dataset == "GDELT":
        cfg.max_history    = 20
        cfg.batch_size     = 32
        cfg.num_epochs     = 30
        cfg.learning_rate  = 5e-4
        cfg.use_reciprocal = True
        cfg.eval_every     = 1
        logger.info("GDELT preset: epochs=30, history=20")

    elif cfg.dataset == "WIKI":
        cfg.max_history    = 10
        cfg.batch_size     = 32
        cfg.num_epochs     = 50
        cfg.learning_rate  = 5e-4
        cfg.use_reciprocal = True
        cfg.eval_every     = 1
        logger.info("WIKI preset: epochs=50, history=10")

    elif cfg.dataset in ("YAGO", "YAGOs"):
        cfg.max_history    = 10
        cfg.batch_size     = 64   # DaeMon: batch_size=64 for YAGO
        cfg.num_epochs     = 50
        cfg.learning_rate  = 5e-4
        cfg.use_reciprocal = True
        cfg.eval_every     = 2
        logger.info(f"{cfg.dataset} preset: epochs=50, history=10, batch=64")

    elif cfg.dataset == "ICEWS18":
        cfg.max_history    = 25
        cfg.batch_size     = 32
        cfg.num_epochs     = 30
        cfg.learning_rate  = 5e-4
        cfg.use_reciprocal = True
        cfg.eval_every     = 1
        logger.info("ICEWS18 preset: epochs=30, history=25")

    elif cfg.dataset == "ICEWS14":
        cfg.max_history    = 25
        cfg.batch_size     = 32
        cfg.num_epochs     = 30
        cfg.learning_rate  = 5e-4
        cfg.use_reciprocal = True
        cfg.eval_every     = 1
        logger.info("ICEWS14 preset: epochs=30, history=25")


def main():
    cfg    = parse_args()
    seed_everything(cfg.seed)
    logger = get_logger("main", cfg.log_dir)

    if not torch.cuda.is_available():
        cfg.device = "cpu"
        cfg.fp16   = False
        logger.warning("CUDA mavjud emas — CPU mode")

    # User override larni preset dan oldin saqlash
    _default       = Config()
    _user_override = {
        k: v for k, v in vars(cfg).items()
        if v != getattr(_default, k, None)
    }

    # Dataset presets
    _preset(cfg, logger)

    # User overrides preset dan ustun
    for k, v in _user_override.items():
        setattr(cfg, k, v)

    logger.info(f"Device: {cfg.device}  |  FP16: {cfg.fp16}")
    logger.info(
        f"Dataset: {cfg.dataset}  |  History: {cfg.max_history}  |  "
        f"Batch: {cfg.batch_size}  |  Epochs: {cfg.num_epochs}  |  LR: {cfg.learning_rate}"
    )

    # ── Load datasets ─────────────────────────────────────────────────────────
    # CHRONOSDataset: entity/relation mappings + filter_dict
    # use_reciprocal=False: SnapshotGraph va trainer o'zi qo'shadi
    ds_kwargs = dict(
        data_dir        = cfg.data_dir,
        dataset         = cfg.dataset,
        num_paths       = 1,    # minimal — NEXUS paths ishlatmaydi
        max_path_len    = 1,
        num_negative    = 1,
        max_history     = 0,    # snapshot history orqali
        use_reciprocal  = False,
    )
    train_ds = CHRONOSDataset(split="train", **ds_kwargs)
    valid_ds = CHRONOSDataset(split="valid", **ds_kwargs)
    test_ds  = CHRONOSDataset(split="test",  **ds_kwargs)

    cfg.num_entities  = train_ds.num_entities
    cfg.num_relations = train_ds.num_relations
    cfg.num_times     = max(train_ds.num_times, valid_ds.num_times, test_ds.num_times)

    num_base_rels = train_ds._base_relations

    logger.info(
        f"|E|={cfg.num_entities}  |R_base|={num_base_rels}  "
        f"|R_total|={num_base_rels*2}  |T|={cfg.num_times}"
    )
    logger.info(
        f"Train:{len(train_ds.quads):,}  "
        f"Valid:{len(valid_ds.quads):,}  "
        f"Test:{len(test_ds.quads):,}"
    )

    # ── Build snapshot graph (train data only) ────────────────────────────────
    snap = SnapshotGraph(
        num_entities   = cfg.num_entities,
        num_relations  = num_base_rels,
        use_reciprocal = cfg.use_reciprocal,
    )
    snap.add_triples(train_ds._load_split("train"))
    logger.info(f"Snapshot timestamps: {len(snap.get_timestamps())}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CHRONOSModel(
        num_entities    = cfg.num_entities,
        num_relations   = num_base_rels,
        num_times       = cfg.num_times,
        entity_dim      = cfg.entity_dim,
        dropout         = cfg.dropout,
        label_smoothing = cfg.label_smoothing,
        max_history     = cfg.max_history,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"NEXUS parametrlar: {n_params/1e6:.2f}M")

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1 and cfg.device == "cuda":
        model = torch.nn.DataParallel(model)
        logger.info(f"DataParallel: {n_gpus} GPU")
    else:
        logger.info("Single GPU/CPU")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = CHRONOSTrainer(
        model          = model,
        cfg            = cfg,
        train_quads    = train_ds.quads,
        valid_quads    = valid_ds.quads,
        test_quads     = test_ds.quads,
        valid_dataset  = valid_ds,
        test_dataset   = test_ds,
        snapshot_graph = snap,
    )

    tm = trainer.fit()
    logger.info(f"Final Test: {tm}")


if __name__ == "__main__":
    main()
