# utils/paths.py
"""
Temporal Knowledge Graph uchun yo'l (path) sampling.

AdjList: entity → [(neighbor, relation, time), ...]
Inverse edges: relation + INV_OFFSET (10_000) bilan belgilanadi.

TEZLASHTIRISH: BFS o'rniga Random Walk ishlatiladi.
  - BFS : O(V + E) — sekin, katta grafda qabul qilib bo'lmaydi
  - RW  : O(num_paths × max_len) — tez, har doim bir xil vaqt
"""
import random
from collections import defaultdict
from typing import Dict, List, Tuple

AdjList = Dict[int, List[Tuple[int, int, int]]]

INV_OFFSET = 10_000   # Inverse relatsiya offseti


def build_graph(quads: List[Tuple[int, int, int, int]]) -> AdjList:
    """
    Quadrupletlardan (s, r, o, t) qo'shnilik listini yasaydi.
    Inverse edges: r + INV_OFFSET.
    """
    adj: AdjList = defaultdict(list)
    for s, r, o, t in quads:
        adj[s].append((o, r,              t))
        adj[o].append((s, r + INV_OFFSET, t))
    return adj


def sample_paths(
    adj: AdjList,
    src: int,
    dst: int,
    t_query: int,
    num_paths: int = 8,
    max_len: int = 3,
) -> List[List[Tuple[int, int, int]]]:
    """
    Random Walk bilan src → dst yo'llarini topadi.

    Har bir urinishda:
      1. src dan tasodifiy qo'shni tanlanadi (t <= t_query)
      2. dst ga yetsa — yo'l topildi
      3. Yetmasa — davom etadi (max_len gacha)

    Bu BFS dan ~100x tezroq: O(num_paths × max_len).
    """
    found: List[List[Tuple[int, int, int]]] = []
    attempts = num_paths * 8   # Ko'proq urinish = ko'proq topilgan yo'l

    for _ in range(attempts):
        if len(found) >= num_paths:
            break

        path: List[Tuple[int, int, int]] = []
        curr    = src
        visited = {src}

        for _ in range(max_len):
            # Temporal constraint: faqat t <= t_query bo'lgan qo'shnilar
            nbrs = [
                (nb, rel, t)
                for nb, rel, t in adj.get(curr, [])
                if t <= t_query and nb not in visited
            ]
            if not nbrs:
                break

            nb, rel, t_edge = random.choice(nbrs)
            path.append((nb, rel, t_edge))
            visited.add(nb)

            if nb == dst:
                found.append(path)
                break
            curr = nb

        # Dst topilmasa ham yo'lni saqlaymiz (qisman yo'l ham foydali)
        if path and (not found or path != found[-1]):
            if nb != dst:   # nb bu oxirgi visited node
                found.append(path)

    return found[:num_paths]


def get_fallback_paths(
    adj: AdjList,
    src: int,
    dst: int,
    t_query: int,
    num_paths: int = 8,
) -> List[List[Tuple[int, int, int]]]:
    """
    Yo'l topilmasa, src dan ketuvchi eng yaqin qo'shnilarni qaytaradi.
    """
    valid = [
        (nb, rel, t)
        for nb, rel, t in adj.get(src, [])
        if t <= t_query
    ]
    random.shuffle(valid)

    paths = [[(nb, rel, t)] for nb, rel, t in valid[:num_paths]]

    if not paths:
        paths = [[(src, 0, t_query)]]

    return paths
