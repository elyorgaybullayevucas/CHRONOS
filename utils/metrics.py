# utils/metrics.py
"""
Filtered ranking metrikalari: MRR, Hits@K.
"""
from typing import Dict, List

import torch


def compute_ranks(
    scores: torch.Tensor,        # (B, num_entities)
    targets: torch.Tensor,       # (B,)
    filter_mask: torch.Tensor,   # (B, num_entities) — True = filtrlansin
    filter_flag: bool = True,
) -> torch.Tensor:
    """
    Har bir misol uchun to'g'ri ob'ektning rangini hisoblaydi.
    filter_mask[i, j] = True  →  entity j i-misol uchun muqobil to'g'ri javob,
    shuning uchun raqobatdan chiqariladi.
    """
    B = scores.size(0)
    if filter_flag:
        # To'g'ri ob'ektning skori saqlab qo'yiladi
        target_scores = scores[torch.arange(B), targets].clone()
        # Boshqa barcha to'g'ri ob'ektlarni minus infga o'rnatamiz
        scores = scores.masked_fill(filter_mask, float("-inf"))
        # To'g'ri ob'ektning skori qayta tiklanadi
        scores[torch.arange(B), targets] = target_scores

    # Rank = (nechta entity skor bo'yicha oldinda) + 1
    ranks = (scores > scores[torch.arange(B), targets].unsqueeze(1)).sum(dim=1) + 1
    return ranks.float()


def ranks_to_metrics(ranks: torch.Tensor, hits_at_k: List[int]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["MRR"] = (1.0 / ranks).mean().item()
    for k in hits_at_k:
        metrics[f"Hits@{k}"] = (ranks <= k).float().mean().item()
    return metrics


def format_metrics(m: Dict[str, float]) -> str:
    parts = [f"MRR={m['MRR']:.4f}"]
    for k, v in m.items():
        if k != "MRR":
            parts.append(f"{k}={v:.4f}")
    return "  ".join(parts)
