# data/dataset.py
"""
CHRONOSDataset:
  - Quadruplet (s, r, o, t) yuklash
  - Temporal path sampling (utils/paths.py orqali)
  - Self-adversarial negative sampling (mixed: random + type-constrained)
  - History: entity s ning t dan oldingi faktlari
  - Filter dict: filtered ranking uchun
"""
import os
import random
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import torch
from torch.utils.data import Dataset

from utils.paths import AdjList, INV_OFFSET, build_graph, get_fallback_paths, sample_paths

Quad = Tuple[int, int, int, int]


class CHRONOSDataset(Dataset):

    def __init__(
        self,
        data_dir:      str,
        dataset:       str,
        split:         str,          # train | valid | test
        num_paths:     int  = 8,
        max_path_len:  int  = 3,
        num_negative:  int  = 256,
        neg_mode:      str  = "mixed",
        max_history:   int  = 0,     # 0 = o'chirilgan
        use_reciprocal: bool = False,
    ):
        super().__init__()
        self.base          = os.path.join(data_dir, dataset)
        self.split         = split
        self.num_paths     = num_paths
        self.max_path_len  = max_path_len
        self.num_negative  = num_negative
        self.neg_mode      = neg_mode
        self.max_history   = max_history
        self.use_reciprocal = use_reciprocal

        # ── Mapping ───────────────────────────────────────────────────────────
        self.ent2id, self.id2ent = self._load_map("entity2id.txt")
        self.rel2id, self.id2rel = self._load_map("relation2id.txt")
        self.num_entities    = len(self.ent2id)
        self._base_relations = len(self.rel2id)
        self.num_relations   = self._base_relations * 2 if use_reciprocal else self._base_relations

        # ── Vaqt soni ─────────────────────────────────────────────────────────
        self.num_times = self._load_num_times()

        # ── Quadrupletlar ─────────────────────────────────────────────────────
        self.all_quads: List[Quad] = self._load_all()
        self.quads:     List[Quad] = self._load_split(split)

        if use_reciprocal and split == "train":
            inv_quads = [
                (o, r + self._base_relations, s, t)
                for s, r, o, t in self.quads
            ]
            self.quads = self.quads + inv_quads
            random.shuffle(self.quads)

        if self.all_quads:
            self.num_times = max(
                self.num_times,
                max(t for _, _, _, t in self.all_quads) + 1,
            )

        # ── Graph ─────────────────────────────────────────────────────────────
        train_q = self._load_split("train")
        if split == "test":
            valid_q = self._load_split("valid")
            graph_q = train_q + valid_q
        else:
            graph_q = train_q
        self.adj: AdjList = build_graph(graph_q)

        # ── Type constraint ───────────────────────────────────────────────────
        self.rel_to_objects: Dict[int, List[int]] = defaultdict(list)
        for s, r, o, t in train_q:
            self.rel_to_objects[r].append(o)

        # ── Filter dict ───────────────────────────────────────────────────────
        # Tail prediction: (s, r, t) → valid objects
        # Head prediction: (o, r_inv, t) → valid subjects  (DaeMon evaluation protocol)
        self.filter_dict: Dict[Tuple, Set[int]] = defaultdict(set)
        for s, r, o, t in self.all_quads:
            self.filter_dict[(s, r, t)].add(o)
            # inverse: head prediction (o, r^{-1}, t) → valid s
            r_inv = r + self._base_relations
            self.filter_dict[(o, r_inv, t)].add(s)

        self.all_ent_ids = list(range(self.num_entities))

    # ── Yordamchi ─────────────────────────────────────────────────────────────

    def _load_map(self, fname: str):
        path = os.path.join(self.base, fname)
        e2id, id2e = {}, {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                p = line.strip().split("\t")
                if len(p) < 2:
                    continue
                try:
                    name, idx = p[0], int(p[1])
                except ValueError:
                    idx, name = int(p[0]), p[1]
                e2id[name] = idx
                id2e[idx]  = name
        return e2id, id2e

    def _load_num_times(self) -> int:
        path = os.path.join(self.base, "stat.txt")
        if not os.path.exists(path):
            return 1
        with open(path) as f:
            parts = f.readline().strip().split()
            return int(parts[2]) if len(parts) >= 3 else 1

    def _load_split(self, split: str) -> List[Quad]:
        path = os.path.join(self.base, f"{split}.txt")
        quads = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                p = line.strip().split("\t")
                if len(p) >= 4:
                    quads.append((int(p[0]), int(p[1]), int(p[2]), int(p[3])))
        return quads

    def _load_all(self) -> List[Quad]:
        all_q = []
        for sp in ("train", "valid", "test"):
            fp = os.path.join(self.base, f"{sp}.txt")
            if os.path.exists(fp):
                all_q.extend(self._load_split(sp))
        return all_q

    # ── Negative sampling ─────────────────────────────────────────────────────

    def _sample_negatives(self, s: int, r: int, o: int, t: int) -> List[int]:
        """Mixed: 50% type-constrained + 50% random."""
        true_objs = self.filter_dict.get((s, r, t), {o})
        negs = set()
        n    = self.num_negative

        if self.neg_mode in ("type", "mixed"):
            pool = self.rel_to_objects.get(r % self._base_relations, self.all_ent_ids)
            cands = [e for e in pool if e not in true_objs]
            if cands:
                k = n // 2 if self.neg_mode == "mixed" else n
                negs.update(random.sample(cands, min(k, len(cands))))

        while len(negs) < n:
            e = random.randint(0, self.num_entities - 1)
            if e not in true_objs:
                negs.add(e)

        return list(negs)[:n]

    # ── Dataset interfeysi ────────────────────────────────────────────────────

    def __len__(self):
        return len(self.quads)

    def __getitem__(self, idx: int) -> Dict:
        s, r, o, t = self.quads[idx]

        # ── Path sampling ─────────────────────────────────────────────────────
        raw_paths = sample_paths(
            self.adj, s, o, t,
            num_paths=self.num_paths,
            max_len=self.max_path_len,
        )
        if not raw_paths:
            raw_paths = get_fallback_paths(self.adj, s, o, t, self.num_paths)

        # Oxirgi hopni olib tashlaymiz (target o ni yashiramiz)
        paths = []
        for path in raw_paths:
            if len(path) > 1:
                paths.append(path[:-1])
            else:
                nbrs = self.adj.get(s, [])
                alt  = [(nb, rl, tm) for nb, rl, tm in nbrs if nb != o and tm <= t]
                if alt:
                    paths.append([random.choice(alt)])
                else:
                    paths.append([(s, 0, t)])

        # Padding
        if not paths:
            paths = [[(s, 0, t)]]
        while len(paths) < self.num_paths:
            paths.append(random.choice(paths))
        paths = paths[:self.num_paths]

        # ── Negatives ─────────────────────────────────────────────────────────
        neg_objs = []
        if self.split == "train":
            neg_objs = self._sample_negatives(s, r, o, t)

        # ── History ───────────────────────────────────────────────────────────
        history = []
        if self.max_history > 0:
            raw_hist = [
                (nb, rl, tm)
                for nb, rl, tm in self.adj.get(s, [])
                if tm < t
            ]
            raw_hist.sort(key=lambda x: -x[2])   # Most-recent-first
            history = raw_hist[:self.max_history]

        return {
            "subject":     s,
            "relation":    r,
            "object":      o,
            "time":        t,
            "paths":       paths,
            "neg_objects": neg_objs,
            "history":     history,
        }

    # ── Filter mask (evaluation) ──────────────────────────────────────────────

    def get_filter_mask(
        self,
        subjects:  torch.Tensor,
        relations: torch.Tensor,
        times:     torch.Tensor,
        targets:   torch.Tensor,
    ) -> torch.Tensor:
        N    = subjects.size(0)
        mask = torch.zeros(N, self.num_entities, dtype=torch.bool)
        for i in range(N):
            key = (int(subjects[i]), int(relations[i]), int(times[i]))
            for obj in self.filter_dict.get(key, set()):
                if obj != int(targets[i]) and 0 <= obj < self.num_entities:
                    mask[i, obj] = True
        return mask
