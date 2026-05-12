# data/datamodule.py
"""
CHRONOSDataModule — DataLoader yasash, collate_fn, setup.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from config import Config
from data.dataset import CHRONOSDataset

MAX_PATH_LEN = 3   # paths tensorlari uchun max uzunlik


def _pad_paths(
    raw_paths: List[List[Tuple[int, int, int]]],
    num_paths: int,
    max_len:   int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    raw_paths: [[hop1, hop2, ...], ...]   hop = (node, rel, time)
    Returns:
      paths_t : (num_paths, max_len, 3)
      mask_t  : (num_paths,)  True = bu yo'l padding
    """
    paths_t  = torch.zeros(num_paths, max_len, 3, dtype=torch.long)
    mask_t   = torch.ones(num_paths, dtype=torch.bool)   # default = padding

    for i, path in enumerate(raw_paths[:num_paths]):
        mask_t[i] = False   # haqiqiy yo'l
        for j, (nb, rel, tm) in enumerate(path[:max_len]):
            paths_t[i, j, 0] = nb
            paths_t[i, j, 1] = rel
            paths_t[i, j, 2] = tm

    return paths_t, mask_t


def _pad_history(
    raw_hist: List[Tuple[int, int, int]],
    max_history: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    raw_hist: [(nb, rel, time), ...]
    Returns:
      hist_t : (max_history, 3)
      mask_t : (max_history,) True = padding
    """
    hist_t = torch.zeros(max_history, 3, dtype=torch.long)
    mask_t = torch.ones(max_history, dtype=torch.bool)

    for i, (nb, rel, tm) in enumerate(raw_hist[:max_history]):
        hist_t[i, 0] = nb
        hist_t[i, 1] = rel
        hist_t[i, 2] = tm
        mask_t[i]    = False

    return hist_t, mask_t


def collate_fn(
    batch:       List[Dict],
    num_negative: int,
    num_paths:    int,
    max_path_len: int,
    max_history:  int,
) -> Dict[str, torch.Tensor]:
    """Custom collate: paths va historyni padding qiladi."""
    B = len(batch)

    subjects    = torch.tensor([b["subject"]  for b in batch], dtype=torch.long)
    relations   = torch.tensor([b["relation"] for b in batch], dtype=torch.long)
    objects     = torch.tensor([b["object"]   for b in batch], dtype=torch.long)
    times       = torch.tensor([b["time"]     for b in batch], dtype=torch.long)

    # Paths
    all_paths  = torch.zeros(B, num_paths, max_path_len, 3, dtype=torch.long)
    all_pmasks = torch.ones(B, num_paths, dtype=torch.bool)
    for i, b in enumerate(batch):
        pt, pm = _pad_paths(b["paths"], num_paths, max_path_len)
        all_paths[i]  = pt
        all_pmasks[i] = pm

    # Negative objects (faqat train da)
    max_neg = max((len(b["neg_objects"]) for b in batch), default=0)
    if max_neg > 0:
        neg_tensor = torch.zeros(B, max_neg, dtype=torch.long)
        for i, b in enumerate(batch):
            neg = b["neg_objects"]
            if neg:
                neg_tensor[i, :len(neg)] = torch.tensor(neg, dtype=torch.long)
    else:
        neg_tensor = torch.zeros(B, 1, dtype=torch.long)

    # History
    if max_history > 0:
        all_hist   = torch.zeros(B, max_history, 3, dtype=torch.long)
        all_hmasks = torch.ones(B, max_history, dtype=torch.bool)
        for i, b in enumerate(batch):
            ht, hm = _pad_history(b["history"], max_history)
            all_hist[i]   = ht
            all_hmasks[i] = hm
    else:
        all_hist   = torch.zeros(B, 1, 3, dtype=torch.long)
        all_hmasks = torch.ones(B, 1, dtype=torch.bool)

    return {
        "subject":     subjects,
        "relation":    relations,
        "object":      objects,
        "time":        times,
        "paths":       all_paths,
        "path_masks":  all_pmasks,
        "neg_objects": neg_tensor,
        "history":     all_hist,
        "hist_mask":   all_hmasks,
    }


class CHRONOSDataModule:

    def __init__(self, cfg: Config):
        self.cfg      = cfg
        self.train_ds: Optional[CHRONOSDataset] = None
        self.valid_ds: Optional[CHRONOSDataset] = None
        self.test_ds:  Optional[CHRONOSDataset] = None

    def setup(self):
        cfg = self.cfg
        kwargs = dict(
            data_dir       = cfg.data_dir,
            dataset        = cfg.dataset,
            num_paths      = cfg.num_paths,
            max_path_len   = cfg.max_path_len,
            num_negative   = cfg.num_negative,
            max_history    = cfg.max_history if cfg.use_history else 0,
            use_reciprocal = cfg.use_reciprocal,
        )
        self.train_ds = CHRONOSDataset(split="train", **kwargs)
        self.valid_ds = CHRONOSDataset(split="valid", **kwargs)
        self.test_ds  = CHRONOSDataset(split="test",  **kwargs)

        # Config ni to'ldirish
        cfg.num_entities  = self.train_ds.num_entities
        cfg.num_relations = self.train_ds.num_relations
        cfg.num_times     = max(
            self.train_ds.num_times,
            self.valid_ds.num_times,
            self.test_ds.num_times,
        )

    def _make_loader(self, ds: CHRONOSDataset, shuffle: bool) -> DataLoader:
        cfg = self.cfg
        fn  = lambda batch: collate_fn(
            batch,
            num_negative  = cfg.num_negative,
            num_paths     = cfg.num_paths,
            max_path_len  = cfg.max_path_len,
            max_history   = cfg.max_history if cfg.use_history else 0,
        )
        return DataLoader(
            ds,
            batch_size  = cfg.batch_size,
            shuffle     = shuffle,
            num_workers = cfg.num_workers,
            collate_fn  = fn,
            pin_memory  = True,
            drop_last   = shuffle,
        )

    def train_loader(self) -> DataLoader:
        return self._make_loader(self.train_ds, shuffle=True)

    def valid_loader(self) -> DataLoader:
        return self._make_loader(self.valid_ds, shuffle=False)

    def test_loader(self) -> DataLoader:
        return self._make_loader(self.test_ds, shuffle=False)
