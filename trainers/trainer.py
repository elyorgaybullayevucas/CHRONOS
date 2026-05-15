# trainers/trainer.py
"""
NEXUSTrainer — DaeMon uslubida timestamp-ordered training + evaluation.

Asosiy farqlar (eski trainer dan):
  - Random triple batching  →  Timestamp-ordered batching (DaeMon kabi)
  - OneCycleLR              →  Adam + cosine decay (DaeMon kabi, lr=5e-4)
  - Faqat tail prediction   →  Head + tail prediction (DaeMon evaluation protocol)
"""
import os
import random
import math
from typing import Dict, List

import torch
import torch.nn as nn

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import Config
from data.snapshot import SnapshotGraph
from data.dataset import CHRONOSDataset
from models.chronos_model import CHRONOSModel
from utils.logging import get_logger
from utils.metrics import compute_ranks, format_metrics, ranks_to_metrics


class CHRONOSTrainer:

    def __init__(
        self,
        model:           CHRONOSModel,
        cfg:             Config,
        train_quads:     list,           # [(s,r,o,t), ...]
        valid_quads:     list,
        test_quads:      list,
        valid_dataset:   CHRONOSDataset,
        test_dataset:    CHRONOSDataset,
        snapshot_graph:  SnapshotGraph,  # train data only
    ):
        self.model         = model
        self._raw          = model.module if hasattr(model, "module") else model
        self.cfg           = cfg
        self.valid_dataset = valid_dataset
        self.test_dataset  = test_dataset
        self.snap          = snapshot_graph

        self.device = torch.device(
            cfg.device if torch.cuda.is_available() else "cpu"
        )
        self._raw.to(self.device)
        self.logger = get_logger("nexus_trainer", cfg.log_dir)

        # ── Group quads by timestamp ──────────────────────────────────────────
        self.train_by_time = SnapshotGraph.group_quads_by_time(train_quads)
        self.valid_by_time = SnapshotGraph.group_quads_by_time(valid_quads)
        self.test_by_time  = SnapshotGraph.group_quads_by_time(test_quads)

        # Add reciprocal triples for training
        if cfg.use_reciprocal:
            R = self._raw.num_base_relations
            for t in list(self.train_by_time.keys()):
                orig = self.train_by_time[t]
                inv  = [(o, r + R, s) for s, r, o in orig]
                self.train_by_time[t] = orig + inv

        # ── Optimizer (DaeMon: Adam lr=5e-4, no scheduler fancy) ─────────────
        self.optimizer = Adam(
            self._raw.parameters(),
            lr           = cfg.learning_rate,
            weight_decay = cfg.weight_decay,
        )

        # Cosine annealing over full training
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max  = cfg.num_epochs,
            eta_min = cfg.learning_rate * 0.01,
        )

        # ── Mixed precision ───────────────────────────────────────────────────
        self.use_fp16 = cfg.fp16 and self.device.type == "cuda"
        if self.use_fp16:
            try:
                self.scaler = GradScaler(device="cuda")
            except TypeError:
                self.scaler = GradScaler()
        else:
            self.scaler = None

        self.best_mrr   = 0.0
        self.best_epoch = 0
        os.makedirs(cfg.save_dir, exist_ok=True)

    # ── Build accumulated snapshot graphs ─────────────────────────────────────

    def _build_valid_snap(self) -> SnapshotGraph:
        """Valid evaluation: history = train snapshots."""
        vs = SnapshotGraph(
            self.snap.num_entities,
            self.snap.num_relations,
            self.snap.use_reciprocal,
        )
        vs._graphs     = dict(self.snap._graphs)
        vs._timestamps = list(self.snap._timestamps)
        return vs

    def _build_test_snap(self) -> SnapshotGraph:
        """Test evaluation: history = train + valid snapshots."""
        ts = SnapshotGraph(
            self.snap.num_entities,
            self.snap.num_relations,
            self.snap.use_reciprocal,
        )
        ts._graphs     = dict(self.snap._graphs)
        ts._timestamps = list(self.snap._timestamps)
        # Add valid triples to history
        valid_raw = []
        for t, triples in self.valid_by_time.items():
            for s, r, o in triples:
                valid_raw.append((s, r, o, t))
        if valid_raw:
            ts.add_triples(valid_raw)
        return ts

    # ── Training ──────────────────────────────────────────────────────────────

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        totals = {"loss": 0.0, "link": 0.0}
        n = 0

        timestamps = sorted(self.train_by_time.keys())
        random.shuffle(timestamps)

        for t in timestamps:
            triples = self.train_by_time[t]
            if not triples:
                continue

            snap_graphs = self.snap.get_history(t, self.cfg.max_history)

            # Batch triples within this timestamp
            random.shuffle(triples)
            for i in range(0, len(triples), self.cfg.batch_size):
                batch = triples[i: i + self.cfg.batch_size]
                if len(batch) < 2:
                    continue

                s = torch.tensor([x[0] for x in batch], dtype=torch.long, device=self.device)
                r = torch.tensor([x[1] for x in batch], dtype=torch.long, device=self.device)
                o = torch.tensor([x[2] for x in batch], dtype=torch.long, device=self.device)
                t_ten = torch.full((len(batch),), t, dtype=torch.long, device=self.device)

                # Self-adversarial negative sampling (DaeMon uslubida)
                B_cur  = s.size(0)
                neg_o  = torch.randint(
                    0, self._raw.num_entities,
                    (B_cur, self._raw.num_negative),
                    dtype=torch.long, device=self.device,
                )

                with autocast(device_type=self.device.type, enabled=self.use_fp16):
                    _, losses = self.model(s, r, o, t_ten, snap_graphs,
                                          neg_objects=neg_o)
                    total_loss = (
                        self.cfg.w_link * losses["link"]
                        + 0.005 * losses.get("ortho", losses["link"].new_tensor(0.0))
                    )

                self.optimizer.zero_grad()
                if self.use_fp16:
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self._raw.parameters(), self.cfg.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self._raw.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()

                totals["loss"] += total_loss.item()
                totals["link"] += losses["link"].item()
                n += 1

        self.scheduler.step()
        lr = self.optimizer.param_groups[0]["lr"]
        return {k: v / max(n, 1) for k, v in totals.items()} | {"lr": lr}

    # ── Evaluation (DaeMon: head + tail prediction) ───────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        eval_by_time:   Dict[int, List],
        filter_dataset: CHRONOSDataset,
        history_snap:   SnapshotGraph,
    ) -> Dict[str, float]:
        """
        Head + tail prediction (DaeMon evaluation protocol).
        Original triple (s, r, o, t) → 2 queries:
          - Tail: (s, r, ?, t)
          - Head: (o, r_inv, ?, t)   r_inv = r + num_base_rels
        """
        self.model.eval()
        all_ranks: List[torch.Tensor] = []
        R = self._raw.num_base_relations

        for t in sorted(eval_by_time.keys()):
            triples = eval_by_time[t]
            if not triples:
                continue

            snap_graphs = history_snap.get_history(t, self.cfg.max_history)

            # Build tail queries + head queries
            tail_queries = [(s, r,     o) for s, r, o in triples]
            head_queries = [(o, r + R, s) for s, r, o in triples]
            all_queries  = tail_queries + head_queries

            for i in range(0, len(all_queries), self.cfg.batch_size):
                batch = all_queries[i: i + self.cfg.batch_size]
                s = torch.tensor([x[0] for x in batch], dtype=torch.long, device=self.device)
                r = torch.tensor([x[1] for x in batch], dtype=torch.long, device=self.device)
                o = torch.tensor([x[2] for x in batch], dtype=torch.long, device=self.device)
                t_ten = torch.full((len(batch),), t, dtype=torch.long, device=self.device)

                scores = self._raw.predict(s, r, t_ten, snap_graphs)

                filter_mask = filter_dataset.get_filter_mask(
                    s.cpu(), r.cpu(), t_ten.cpu(), o.cpu()
                ).to(self.device)

                ranks = compute_ranks(scores, o, filter_mask, self.cfg.filter_flag)
                all_ranks.append(ranks.float().cpu())

        if not all_ranks:
            return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}

        return ranks_to_metrics(torch.cat(all_ranks), self.cfg.hits_at_k)

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save(self, epoch: int, metrics: Dict, tag: str = "best"):
        path = os.path.join(self.cfg.save_dir, f"{self.cfg.dataset}_{tag}.pt")
        torch.save({
            "epoch":       epoch,
            "model_state": self._raw.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "metrics":     metrics,
            "config":      self.cfg.__dict__,
        }, path)
        self.logger.info(f"  → Checkpoint: {path}")

    def load(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self._raw.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self.logger.info(f"Yuklandi: {path}  (epoch {ckpt['epoch']})")
        return ckpt["epoch"]

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self) -> Dict[str, float]:
        start = 0
        if self.cfg.resume:
            start = self.load(self.cfg.resume) + 1

        self.logger.info("=" * 70)
        self.logger.info(
            f"  NEXUS TRAINING: {self.cfg.dataset}  |  "
            f"Epochs: {self.cfg.num_epochs}  |  FP16: {self.use_fp16}  |  "
            f"History: {self.cfg.max_history}  |  Batch: {self.cfg.batch_size}"
        )
        self.logger.info("=" * 70)

        valid_snap = self._build_valid_snap()
        test_snap  = self._build_test_snap()

        for epoch in range(start, self.cfg.num_epochs):
            tr = self.train_one_epoch(epoch)
            self.logger.info(
                f"Epoch {epoch+1:3d}/{self.cfg.num_epochs} | "
                f"Loss:{tr['loss']:.4f}  Link:{tr['link']:.4f}  LR:{tr['lr']:.2e}"
            )

            if (epoch + 1) % self.cfg.eval_every == 0:
                vm = self.evaluate(self.valid_by_time, self.valid_dataset, valid_snap)
                self.logger.info(f"  [Valid] {format_metrics(vm)}")

                if vm["MRR"] > self.best_mrr:
                    self.best_mrr   = vm["MRR"]
                    self.best_epoch = epoch + 1
                    self.save(epoch + 1, vm, "best")
                    self.logger.info(f"  ★ Yangi rekord! MRR={self.best_mrr:.4f}")

        # ── Test ─────────────────────────────────────────────────────────────
        self.logger.info("=" * 70)
        best_path = os.path.join(self.cfg.save_dir, f"{self.cfg.dataset}_best.pt")
        if os.path.exists(best_path):
            self.load(best_path)

        tm = self.evaluate(self.test_by_time, self.test_dataset, test_snap)
        self.logger.info(f"[TEST]  {format_metrics(tm)}")
        self.logger.info(f"Best epoch: {self.best_epoch}  Best valid MRR: {self.best_mrr:.4f}")
        self.logger.info("=" * 70)
        return tm
