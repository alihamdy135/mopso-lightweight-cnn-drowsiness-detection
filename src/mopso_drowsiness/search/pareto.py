from ..config import *

def _objectives_from_info(info: Dict[str, Any]) -> np.ndarray:
    """Lower-is-better 5-vector used by NSGA-II to rank a population by Pareto
    dominance."""
    return np.array([
        -info.get("accuracy", 0.0),
        float(info.get("n_params", np.inf)),
        float(info.get("flops", np.inf)),
        float(info.get("inference_ms", np.inf)),
        float(info.get("peak_memory_mb", np.inf)),
    ], dtype=float)


def _pareto_dominates(objs1: np.ndarray, objs2: np.ndarray) -> bool:
    """True if objs1 dominates objs2 (lower is better in every dimension)."""
    return bool(np.all(objs1 <= objs2) and np.any(objs1 < objs2))


def fast_non_dominated_sort(objs_list: List[np.ndarray]) -> List[List[int]]:
    """Deb et al. (2002) Algorithm: partitions `objs_list` (lower-is-better vectors) into
    Pareto fronts. Returns a list of fronts, each a list of indices into objs_list, with
    fronts[0] = the non-dominated set (rank 0), fronts[1] = next rank, etc."""
    n = len(objs_list)
    dominated_by = [[] for _ in range(n)]
    dom_count = [0] * n
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _pareto_dominates(objs_list[p], objs_list[q]):
                dominated_by[p].append(q)
            elif _pareto_dominates(objs_list[q], objs_list[p]):
                dom_count[p] += 1
        if dom_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_by[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()
    return fronts


def crowding_distance(front_indices: List[int], objs_list: List[np.ndarray]) -> Dict[int, float]:
    """Deb et al. (2002) crowding-distance: for one front, returns {index: distance},
    with boundary solutions on each objective getting infinite distance (always kept)."""
    distance = {idx: 0.0 for idx in front_indices}
    l = len(front_indices)
    if l == 0:
        return distance

    n_obj = len(objs_list[0])
    for m in range(n_obj):
        front_sorted = sorted(front_indices, key=lambda idx: objs_list[idx][m])
        distance[front_sorted[0]] = np.inf
        distance[front_sorted[-1]] = np.inf
        obj_min = objs_list[front_sorted[0]][m]
        obj_max = objs_list[front_sorted[-1]][m]
        if obj_max - obj_min < 1e-12:
            continue
        for k in range(1, l - 1):
            distance[front_sorted[k]] += (
                (objs_list[front_sorted[k + 1]][m] - objs_list[front_sorted[k - 1]][m])
                / (obj_max - obj_min)
            )
    return distance


def nsga2_tournament(indices: List[int], ranks: Dict[int, int], crowd: Dict[int, float],
                      k: int = 2) -> int:
    """Binary tournament selection by (Pareto rank ascending, crowding distance descending)
    -- the standard NSGA-II crowded-comparison operator."""
    candidates = random.sample(indices, min(k, len(indices)))
    best = candidates[0]
    for c in candidates[1:]:
        if (ranks[c], -crowd[c]) < (ranks[best], -crowd[best]):
            best = c
    return best
