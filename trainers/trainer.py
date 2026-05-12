# trainers/trainer.py
"""
CHRONOSTrainer — Mixed precision, OneCycleLR, PCL + OrthoReg.
"""
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn

try:
    from torch.amp import GradScaler, autocast
    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _AMP_DEVICE = "cuda"

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from config import Config
from data.dataset import CHRONOSDataset
from models.chronos_model import CHRONOSModel
from utils.logging import get_logger
from utils.metrics import compute_ranks, format_metrics, ranks_to_metrics


class CHRONOSTrainer:

    def __init__(
        self,
        model:         CHRONOSModel,
        cfg:           Config,
        train_loader:  DataLoader,
        valid_loader:  DataLoader,
        test_loader:   DataLoader,
        valid_dataset: CHRONOSDataset,
        test_dataset:  CHRONOSDataset,
    ):
        self.model   = model                                   # DataParallel bo'lishi mumkin
        self._raw    = model.module if hasattr(model, "module") else model
        self.cfg     = cfg
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader  = test_loader
        self.valid_dataset = valid_dataset
        self.test_dataset  = test_dataset

        self.device = torch.device(
            cfg.device if torch.cuda.is_available() else "cpu"
        )
        self._raw.to(self.device)
        self.logger = get_logger("chronos_trainer", cfg.log_dir)

        # ── Optimizer: 3-guruhli differentsial LR ─────────────────────────────
        # Guruh 1: entity + relation embeddinglari (eng sekin)
        emb_params: list = []
        for name in ("ent_emb", "rel_emb"):
            m = getattr(self._raw, name, None)
            if m is not None:
                emb_params += list(m.parameters())

        # Guruh 2: RPE, TFE, TPL, history, QATS, TRTM (o'rta)
        extra_params: list = []
        for name in ("rpe", "tfe", "tpl", "hist_encoder", "qats", "trtm",
                     "trtm_proj", "hist_gate", "hist_norm", "path_encoder",
                     "path_agg", "direct_head", "pcl", "rel_temp"):
            m = getattr(self._raw, name, None)
            if m is not None:
                extra_params += list(m.parameters())

        # Guruh 3: fusion, scorer, ent_proj (to'liq LR)
        emb_set   = {id(p) for p in emb_params}
        extra_set = {id(p) for p in extra_params}
        other_params = [
            p for p in self._raw.parameters()
            if id(p) not in emb_set and id(p) not in extra_set
        ]

        self.optimizer = AdamW([
            {"params": emb_params,   "lr": cfg.learning_rate * 0.1},
            {"params": extra_params, "lr": cfg.learning_rate * 0.5},
            {"params": other_params, "lr": cfg.learning_rate},
        ], weight_decay=cfg.weight_decay)

        # ── OneCycleLR scheduler ──────────────────────────────────────────────
        steps_per_epoch = len(train_loader)
        total_steps     = cfg.num_epochs * steps_per_epoch
        base_lr         = cfg.learning_rate
        max_lrs         = [base_lr * 0.1, base_lr * 0.5, base_lr]

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr         = max_lrs,
            total_steps    = total_steps,
            pct_start      = 0.05,
            anneal_strategy = "cos",
            div_factor     = 25,
            final_div_factor = 1e4,
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

    # ── Training ──────────────────────────────────────────────────────────────

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = {
            "loss": 0, "link": 0, "self_adv": 0, "pcl": 0, "ortho": 0
        }
        n = 0

        for batch in self.train_loader:
            subjects    = batch["subject"].to(self.device)
            relations   = batch["relation"].to(self.device)
            objects     = batch["object"].to(self.device)
            times       = batch["time"].to(self.device)
            paths       = batch["paths"].to(self.device)
            path_masks  = batch["path_masks"].to(self.device)
            neg_objects = batch["neg_objects"].to(self.device)
            history     = batch["history"].to(self.device)
            hist_mask   = batch["hist_mask"].to(self.device)

            with autocast(device_type="cuda", enabled=self.use_fp16):
                scores, losses = self.model(
                    subjects, relations, objects, times,
                    paths, path_masks, neg_objects,
                    history=history, hist_mask=hist_mask,
                )
                total_loss = (
                    self.cfg.w_link     * losses["link"]
                    + self.cfg.w_self_adv * losses["self_adv"]
                    + self.cfg.w_pcl      * losses["pcl"]
                    + self.cfg.w_ortho_reg * losses["ortho_reg"]
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

            self.scheduler.step()

            totals["loss"]     += total_loss.item()
            totals["link"]     += losses["link"].item()
            totals["self_adv"] += losses["self_adv"].item()
            totals["pcl"]      += losses["pcl"].item()
            totals["ortho"]    += losses["ortho_reg"].item()
            n += 1

        lr = self.optimizer.param_groups[-1]["lr"]
        return {k: v / max(n, 1) for k, v in totals.items()} | {"lr": lr}

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        loader:  DataLoader,
        dataset: CHRONOSDataset,
    ) -> Dict[str, float]:
        self.model.eval()
        all_ranks: List[torch.Tensor] = []

        for batch in loader:
            subjects   = batch["subject"].to(self.device)
            relations  = batch["relation"].to(self.device)
            objects    = batch["object"].to(self.device)
            times      = batch["time"].to(self.device)
            paths      = batch["paths"].to(self.device)
            path_masks = batch["path_masks"].to(self.device)
            history    = batch["history"].to(self.device)
            hist_mask  = batch["hist_mask"].to(self.device)

            with autocast(device_type="cuda", enabled=self.use_fp16):
                scores = self._raw.predict(
                    subjects, relations, times, paths, path_masks,
                    history=history, hist_mask=hist_mask,
                )

            filter_mask = dataset.get_filter_mask(
                subjects.cpu(), relations.cpu(),
                times.cpu(), objects.cpu(),
            ).to(self.device)

            ranks = compute_ranks(
                scores, objects, filter_mask,
                filter_flag=self.cfg.filter_flag,
            )
            all_ranks.append(ranks.float().cpu())

        all_ranks_t = torch.cat(all_ranks)
        return ranks_to_metrics(all_ranks_t, self.cfg.hits_at_k)

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save(self, epoch: int, metrics: Dict, tag: str = "best"):
        path = os.path.join(self.cfg.save_dir, f"{self.cfg.dataset}_{tag}.pt")
        torch.save({
            "epoch":       epoch,
            "model_state": self._raw.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "sched_state": self.scheduler.state_dict(),
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

    # ── To'liq o'qitish ───────────────────────────────────────────────────────

    def fit(self) -> Dict[str, float]:
        start = 0
        if self.cfg.resume:
            start = self.load(self.cfg.resume) + 1

        self.logger.info("=" * 72)
        self.logger.info(f"  CHRONOS O'QITISH: {self.cfg.dataset}  |  "
                         f"Epochlar: {self.cfg.num_epochs}  |  "
                         f"FP16: {self.use_fp16}")
        self.logger.info("=" * 72)

        for epoch in range(start, self.cfg.num_epochs):
            tr = self.train_one_epoch(epoch)
            self.logger.info(
                f"Epoch {epoch+1:3d}/{self.cfg.num_epochs} | "
                f"Loss:{tr['loss']:.4f} | "
                f"Link:{tr['link']:.4f} | "
                f"Adv:{tr['self_adv']:.4f} | "
                f"PCL:{tr['pcl']:.4f} | "
                f"Ortho:{tr['ortho']:.3f} | "
                f"LR:{tr['lr']:.2e}"
            )

            if (epoch + 1) % self.cfg.eval_every == 0:
                vm = self.evaluate(self.valid_loader, self.valid_dataset)
                self.logger.info(f"  [Valid] {format_metrics(vm)}")

                if vm["MRR"] > self.best_mrr:
                    self.best_mrr   = vm["MRR"]
                    self.best_epoch = epoch + 1
                    self.save(epoch + 1, vm, "best")
                    self.logger.info(f"  ★ Yangi rekord! MRR={self.best_mrr:.4f}")

        # Test
        self.logger.info("=" * 72)
        self.logger.info(f"Eng yaxshi epoch: {self.best_epoch}  |  "
                         f"Valid MRR: {self.best_mrr:.4f}")
        best_path = os.path.join(self.cfg.save_dir, f"{self.cfg.dataset}_best.pt")
        if os.path.exists(best_path):
            self.load(best_path)
        tm = self.evaluate(self.test_loader, self.test_dataset)
        self.logger.info(f"[TEST]  {format_metrics(tm)}")
        self.logger.info("=" * 72)
        return tm
