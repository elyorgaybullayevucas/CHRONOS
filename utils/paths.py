# utils/paths.py
"""
Temporal Knowledge Graph uchun yo'l (path) sampling.

AdjList: entity → [(neighbor, relation, time), ...]
Inverse edges: relation + INV_OFFSET (10_000) bilan belgilanadi.
"""
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

AdjList = Dict[int, List[Tuple[int, int, int]]]

INV_OFFSET = 10_000   # Inverse relatsiya offseti


def build_graph(quads: List[Tuple[int, int, int, int]]) -> AdjList:
    """
    Quadrupletlar ro'yxatidan (s, r, o, t) yo'naltirilmagan qo'shnilik listini yasaydi.
    Har bir qirraga teskari (inverse) qirrasi ham qo'shiladi.
    Inverse relatsiya indeksi = r + INV_OFFSET.
    """
    adj: AdjList = defaultdict(list)
    for s, r, o, t in quads:
        adj[s].append((o, r,              t))   # To'g'ri yo'nalish
        adj[o].append((s, r + INV_OFFSET, t))   # Teskari yo'nalish
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
    src → dst ga boruvchi yo'llarni BFS/random walk bilan topadi.
    Faqat t <= t_query bo'lgan qirralar ishlatiladi (temporal constraint).

    Har bir yo'l: [(node, relation, time), ...] — har bir hop.
    """
    found: List[List[Tuple[int, int, int]]] = []
    # BFS queue: (current_node, path_so_far, visited_set)
    queue = [(src, [], {src})]

    while queue and len(found) < num_paths * 3:
        curr, path, visited = queue.pop(0)
        if len(path) >= max_len:
            continue

        nbrs = adj.get(curr, [])
        random.shuffle(nbrs)

        for nb, rel, t_edge in nbrs:
            if t_edge > t_query:
                continue
            if nb in visited:
                continue

            new_path = path + [(nb, rel, t_edge)]
            new_visited = visited | {nb}

            if nb == dst:
                found.append(new_path)
                if len(found) >= num_paths * 3:
                    break
            else:
                queue.append((nb, new_path, new_visited))

    # Tasodifiy num_paths ta tanlaymiz
    if len(found) > num_paths:
        found = random.sample(found, num_paths)
    return found


def get_fallback_paths(
    adj: AdjList,
    src: int,
    dst: int,
    t_query: int,
    num_paths: int = 8,
) -> List[List[Tuple[int, int, int]]]:
    """
    dst topilmasa, src dan ketuvchi yo'llarni fallback sifatida qaytaradi.
    """
    paths = []
    src_nbrs = adj.get(src, [])
    valid = [(nb, rel, t) for nb, rel, t in src_nbrs if t <= t_query]
    random.shuffle(valid)

    for nb, rel, t_edge in valid[:num_paths]:
        paths.append([(nb, rel, t_edge)])

    if not paths:
        paths = [[(src, 0, t_query)]]

    return paths
