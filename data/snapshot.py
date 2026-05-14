# data/snapshot.py
"""
SnapshotGraph — TKG ma'lumotlarini timestamp bo'yicha tashkil qilish.

DaeMon uslubida har bir timestamp uchun graf (src, rel, dst) tensorlarini saqlaydi.
Model training va evaluation da ishlatiladi.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# (src_tensor, rel_tensor, dst_tensor)
SnapGraph = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class SnapshotGraph:
    """
    TKG snapshot manager.

    Barcha tripllarni timestamp bo'yicha guruhlaydi.
    Model uchun "oxirgi K snapshot grafini qaytaradi" degan funksiya bor.
    """

    def __init__(
        self,
        num_entities:   int,
        num_relations:  int,           # base, inversesiz
        use_reciprocal: bool = True,
    ):
        self.num_entities   = num_entities
        self.num_relations  = num_relations  # base count
        self.total_rels     = num_relations * 2
        self.use_reciprocal = use_reciprocal

        self._graphs: Dict[int, SnapGraph] = {}
        self._timestamps: List[int] = []

    # ── Data ingestion ────────────────────────────────────────────────────────

    def add_triples(self, triples):
        """
        triples: iterable of (s, r, o, t) ints.
        Reciprocal (o, r+R, s, t) larini avtomatik qo'shadi.
        """
        snap: Dict[int, list] = defaultdict(list)
        for s, r, o, t in triples:
            snap[int(t)].append((int(s), int(r), int(o)))
            if self.use_reciprocal:
                snap[int(t)].append((int(o), int(r) + self.num_relations, int(s)))

        for t, edges in snap.items():
            src = torch.tensor([e[0] for e in edges], dtype=torch.long)
            rel = torch.tensor([e[1] for e in edges], dtype=torch.long)
            dst = torch.tensor([e[2] for e in edges], dtype=torch.long)

            if t in self._graphs:
                # merge
                old = self._graphs[t]
                src = torch.cat([old[0], src])
                rel = torch.cat([old[1], rel])
                dst = torch.cat([old[2], dst])

            self._graphs[t] = (src, rel, dst)

        self._timestamps = sorted(self._graphs.keys())

    # ── History retrieval ─────────────────────────────────────────────────────

    def get_history(self, t_query: int, history_len: int) -> List[SnapGraph]:
        """t_query dan oldingi oxirgi history_len snapshot grafini qaytaradi."""
        past = [t for t in self._timestamps if t < t_query]
        past = past[-history_len:]
        return [self._graphs[t] for t in past]

    def get_timestamps(self) -> List[int]:
        return self._timestamps

    def get_triples_at(self, t: int):
        """(src, rel, dst) tensors for a given timestamp."""
        if t not in self._graphs:
            return None
        return self._graphs[t]

    # ── Quads by timestamp ────────────────────────────────────────────────────

    @staticmethod
    def group_quads_by_time(quads) -> Dict[int, List[Tuple[int, int, int]]]:
        """(s, r, o, t) list → {t: [(s, r, o), ...]}"""
        d: Dict[int, List] = defaultdict(list)
        for s, r, o, t in quads:
            d[int(t)].append((int(s), int(r), int(o)))
        return d
