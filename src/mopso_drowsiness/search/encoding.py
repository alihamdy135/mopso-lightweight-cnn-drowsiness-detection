from ..config import *

GA_DIM_CHOICES = {
    "filters":      SS_FILTERS,
    "kernel":       SS_KERNELS,
    "activation":   SS_ACTIVATIONS,
    "dropout":      SS_DROPOUT_CONV,
    "bn_placement": SS_BN_PLACEMENT,
}
GA_MAX_DEPTH   = max(SS_DEPTHS)


GA_BLOCK_DIMS  = list(GA_DIM_CHOICES.keys())
GA_N_GENES     = 1 + GA_MAX_DEPTH * len(GA_BLOCK_DIMS) + 3


GA_GENE_RANGES = [len(SS_DEPTHS)]
for _ in range(GA_MAX_DEPTH):
    for dim_name in GA_BLOCK_DIMS:
        GA_GENE_RANGES.append(len(GA_DIM_CHOICES[dim_name]))
GA_GENE_RANGES += [len(SS_DENSE_UNITS), len(SS_DROPOUT_DENSE), len(SS_ACTIVATIONS)]

PSO_N_DIMS = GA_N_GENES


def _clip_index(value: float, n_choices: int) -> int:
    """Rounds a continuous coordinate down to a valid discrete index, clipped to range."""
    idx = int(np.floor(value))
    return max(0, min(idx, n_choices - 1))


def position_to_genome(pos: np.ndarray) -> Dict[str, Any]:
    i = 0
    depth = SS_DEPTHS[_clip_index(pos[i], len(SS_DEPTHS))]; i += 1

    blocks = []
    for b in range(GA_MAX_DEPTH):
        blk = {}
        for dim_name in GA_BLOCK_DIMS:
            choices = GA_DIM_CHOICES[dim_name]
            blk[dim_name] = choices[_clip_index(pos[i], len(choices))]
            i += 1
        if b < depth:
            blocks.append(blk)

    dense_units = SS_DENSE_UNITS[_clip_index(pos[i], len(SS_DENSE_UNITS))]; i += 1
    dense_drop  = SS_DROPOUT_DENSE[_clip_index(pos[i], len(SS_DROPOUT_DENSE))]; i += 1
    dense_act   = SS_ACTIVATIONS[_clip_index(pos[i], len(SS_ACTIVATIONS))]; i += 1

    return {"depth": depth, "blocks": blocks, "dense_units": dense_units,
            "dense_drop": dense_drop, "dense_act": dense_act}


def random_particle_position() -> np.ndarray:
    """Samples a random continuous position, one uniform draw per dimension range."""
    return np.array([random.uniform(0, n) for n in GA_GENE_RANGES], dtype=float)


def random_chromosome() -> np.ndarray:
    """Samples a random discrete chromosome: one uniform-random valid index per gene."""
    return np.array([random.randrange(n) for n in GA_GENE_RANGES], dtype=int)


def chromosome_to_genome(chrom: np.ndarray) -> Dict[str, Any]:
    """Identical decoding logic to position_to_genome, but genes are already valid integer
    indices (no floor/clip needed) -- kept as a separate function so GA's discrete
    representation is explicit and never silently reuses the continuous PSO path."""
    i = 0
    depth = SS_DEPTHS[int(chrom[i]) % len(SS_DEPTHS)]; i += 1

    blocks = []
    for b in range(GA_MAX_DEPTH):
        blk = {}
        for dim_name in GA_BLOCK_DIMS:
            choices = GA_DIM_CHOICES[dim_name]
            blk[dim_name] = choices[int(chrom[i]) % len(choices)]
            i += 1
        if b < depth:
            blocks.append(blk)

    dense_units = SS_DENSE_UNITS[int(chrom[i]) % len(SS_DENSE_UNITS)]; i += 1
    dense_drop  = SS_DROPOUT_DENSE[int(chrom[i]) % len(SS_DROPOUT_DENSE)]; i += 1
    dense_act   = SS_ACTIVATIONS[int(chrom[i]) % len(SS_ACTIVATIONS)]; i += 1

    return {"depth": depth, "blocks": blocks, "dense_units": dense_units,
            "dense_drop": dense_drop, "dense_act": dense_act}


def ga_tournament_select(chromosomes: List[np.ndarray], fitnesses: List[float],
                          k: int = GA_TOURNAMENT_K) -> np.ndarray:
    """Tournament selection: sample k individuals uniformly at random, return the fittest."""
    idxs = random.sample(range(len(chromosomes)), min(k, len(chromosomes)))
    best_idx = max(idxs, key=lambda i: fitnesses[i])
    return chromosomes[best_idx].copy()


def ga_uniform_crossover(parent_a: np.ndarray, parent_b: np.ndarray,
                          crossover_rate: float = GA_CROSSOVER_RATE) -> Tuple[np.ndarray, np.ndarray]:
    """Per-gene uniform crossover: each gene is independently swapped between the two
    children with probability 0.5, applied only if a crossover event fires at all
    (probability `crossover_rate`); otherwise both parents pass through unchanged."""
    if random.random() > crossover_rate:
        return parent_a.copy(), parent_b.copy()

    child_a, child_b = parent_a.copy(), parent_b.copy()
    for gene_idx in range(len(parent_a)):
        if random.random() < 0.5:
            child_a[gene_idx], child_b[gene_idx] = child_b[gene_idx], child_a[gene_idx]
    return child_a, child_b


def ga_mutate(chrom: np.ndarray, mut_prob: float = GA_MUT_PROB) -> np.ndarray:
    """Per-gene mutation: with probability `mut_prob`, resample that gene to a new random
    valid index drawn uniformly from its own discrete choice set (GA_GENE_RANGES)."""
    mutated = chrom.copy()
    for gene_idx, n_choices in enumerate(GA_GENE_RANGES):
        if random.random() < mut_prob:
            mutated[gene_idx] = random.randrange(n_choices)
    return mutated
