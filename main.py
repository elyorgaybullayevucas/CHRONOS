# main.py
"""
AURORA/CHRONOS — ishga tushirish.

Ishlatish:
    python main.py --dataset YAGO
    python main.py --dataset WIKI
    python main.py --dataset ICEWS18
    python main.py --dataset ICEWS14
    python main.py --dataset GDELT
    python main.py --resume checkpoints/YAGO_best.pt
"""
import argparse
import os
import random

import numpy as np
import torch

from config import Config
from data.datamodule import CHRONOSDataModule
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
    p = argparse.ArgumentParser("AURORA — Adaptive Unified Representation for TKG")

    p.add_argument("--dataset",  default="ICEWS18",
                   choices=["ICEWS14", "ICEWS18", "WIKI", "YAGO", "YAGOs", "GDELT"])
    p.add_argument("--data_dir", default="data")
    p.add_argument("--entity_dim",   type=int,   default=256)
    p.add_argument("--relation_dim", type=int,   default=256)
    p.add_argument("--hidden_dim",   type=int,   default=512)
    p.add_argument("--delta_dim",    type=int,   default=64)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--num_paths",    type=int,   default=6)
    p.add_argument("--max_path_len", type=int,   default=3)
    p.add_argument("--num_negative", type=int,   default=256)
    p.add_argument("--batch_size",      type=int,   default=512)
    p.add_argument("--epochs",          type=int,   default=50,   dest="num_epochs")
    p.add_argument("--lr",              type=float, default=3e-4, dest="learning_rate")
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--w_link",          type=float, default=1.0)
    p.add_argument("--w_self_adv",      type=float, default=0.5)
    p.add_argument("--use_history",   action="store_true")
    p.add_argument("--max_history",   type=int, default=32)
    p.add_argument("--use_reciprocal", action="store_true")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--save_dir",    default="checkpoints")
    p.add_argument("--log_dir",     default="logs")
    p.add_argument("--resume",      default=None)
    p.add_argument("--no_fp16",     action="store_true")
    p.add_argument("--eval_every",  type=int, default=1)

    args = p.parse_args()
    cfg  = Config()
    for k, v in vars(args).items():
        if k == "no_fp16":
            cfg.fp16 = not v
        elif k in ("use_history", "use_reciprocal") and v:
            setattr(cfg, k, True)
        elif hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _preset(cfg: Config, logger) -> None:
    if cfg.dataset == "GDELT":
        cfg.num_paths      = 4
        cfg.max_path_len   = 2
        cfg.batch_size     = 512
        cfg.num_negative   = 256
        cfg.use_history    = True
        cfg.max_history    = 32
        cfg.use_reciprocal = True
        cfg.w_self_adv     = 0.5
        cfg.dropout        = 0.1
        cfg.learning_rate  = 3e-4
        cfg.num_epochs     = 30
        cfg.num_workers    = 8
        logger.info("GDELT: epochs=30, num_paths=4")

    elif cfg.dataset in ("WIKI", "YAGO", "YAGOs"):
        cfg.num_paths      = 6
        cfg.max_path_len   = 3
        cfg.batch_size     = 512
        cfg.num_negative   = 256
        cfg.use_history    = True
        cfg.max_history    = 32
        cfg.use_reciprocal = True
        cfg.w_self_adv     = 0.5
        cfg.dropout        = 0.1
        cfg.label_smoothing= 0.1
        cfg.learning_rate  = 3e-4
        cfg.num_epochs     = 500 if cfg.dataset in ("YAGO", "YAGOs") else 150
        cfg.num_workers    = 8
        logger.info(f"{cfg.dataset}: epochs={cfg.num_epochs}, batch=512, history=32")

    elif cfg.dataset in ("ICEWS18", "ICEWS14"):
        cfg.num_paths      = 6
        cfg.max_path_len   = 3
        cfg.batch_size     = 512
        cfg.num_negative   = 256
        cfg.use_history    = True
        cfg.max_history    = 32
        cfg.use_reciprocal = True
        cfg.w_self_adv     = 0.5
        cfg.dropout        = 0.1
        cfg.learning_rate  = 3e-4
        cfg.num_epochs     = 50
        cfg.num_workers    = 8
        logger.info(f"{cfg.dataset}: epochs=50, batch=512")


def main():
    cfg    = parse_args()
    seed_everything(cfg.seed)
    logger = get_logger("main", cfg.log_dir)

    if not torch.cuda.is_available():
        cfg.device = "cpu"
        cfg.fp16   = False
        logger.warning("CUDA mavjud emas — CPU")

    logger.info(f"Device: {cfg.device}  |  FP16: {cfg.fp16}")

    # Dataset sozlamalari dm.setup() DAN OLDIN
    _preset(cfg, logger)

    logger.info(f"Dataset yuklanmoqda: {cfg.dataset} (reciprocal={cfg.use_reciprocal})")
    dm = CHRONOSDataModule(cfg)
    dm.setup()

    logger.info(f"Dataset={cfg.dataset} | |E|={cfg.num_entities} | "
                f"|R|={cfg.num_relations} | |T|={cfg.num_times}")
    logger.info(f"Train:{len(dm.train_ds):,}  Valid:{len(dm.valid_ds):,}  "
                f"Test:{len(dm.test_ds):,}")

    # num_relations = base (model ichida *2 qilinadi)
    num_base_rels = dm.train_ds._base_relations

    model = CHRONOSModel(
        num_entities    = cfg.num_entities,
        num_relations   = num_base_rels,
        num_times       = cfg.num_times,
        entity_dim      = cfg.entity_dim,
        relation_dim    = cfg.relation_dim,
        hidden_dim      = cfg.hidden_dim,
        delta_dim       = cfg.delta_dim,
        dropout         = cfg.dropout,
        label_smoothing = cfg.label_smoothing,
        use_history     = cfg.use_history,
        max_history     = cfg.max_history,
    )

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"AURORA parametrlar: {n/1e6:.2f}M")

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1 and cfg.device == "cuda":
        model = torch.nn.DataParallel(model)
        logger.info(f"DataParallel: {n_gpus} GPU")
    else:
        logger.info("Single GPU/CPU")

    trainer = CHRONOSTrainer(
        model         = model,
        cfg           = cfg,
        train_loader  = dm.train_loader(),
        valid_loader  = dm.valid_loader(),
        test_loader   = dm.test_loader(),
        valid_dataset = dm.valid_ds,
        test_dataset  = dm.test_ds,
    )

    tm = trainer.fit()
    logger.info(f"Test: {tm}")


if __name__ == "__main__":
    main()
