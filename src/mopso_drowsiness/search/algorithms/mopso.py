from ...config import *
from ...seeding import set_all_seeds
from ..encoding import GA_GENE_RANGES, PSO_N_DIMS, position_to_genome, random_particle_position
from ..objectives import FITNESS_W_ACC, FITNESS_W_FLOPS, FITNESS_W_LATENCY, FITNESS_W_MEMORY, FITNESS_W_PARAMS, compute_population_fitness, evaluate_individual_multiobjective

GA_SWARM_SIZE   = GA_POP_SIZE

GA_ARCHIVE_SIZE = 12
GA_N_DIVISIONS  = 8

PSO_W, PSO_C1, PSO_C2 = 0.6, 1.5, 1.5


def _save_genome(genome, fitness, path):
    """Checkpoints the current best genome + preference fitness to disk, so a long MOPSO
    run can be inspected even if the Kaggle session is killed before it finishes."""
    with open(path, "w") as fh:
        json.dump({"genome": genome, "fitness": fitness}, fh, indent=2)


class MOPSOArchive:
    """External Pareto archive: keeps only mutually non-dominated solutions, using an
    adaptive hypercube grid (over the 5 raw objectives) to pick a diversity-preserving
    leader for each particle and to prune the archive once it overflows."""

    N_OBJ = 5

    def __init__(self, max_size: int = GA_ARCHIVE_SIZE, n_divisions: int = GA_N_DIVISIONS):
        self.max_size = max_size
        self.n_divisions = n_divisions
        self.positions: List[np.ndarray] = []
        self.genomes: List[Dict[str, Any]] = []
        self.objectives: List[np.ndarray] = []
        self.infos: List[Dict[str, Any]] = []

    @staticmethod
    def _to_objectives(info: Dict[str, Any]) -> np.ndarray:
        return np.array([
            -info.get("accuracy", 0.0),
            float(info.get("n_params", np.inf)),
            float(info.get("flops", np.inf)),
            float(info.get("inference_ms", np.inf)),
            float(info.get("peak_memory_mb", np.inf)),
        ], dtype=float)

    def _dominates(self, objs1: np.ndarray, objs2: np.ndarray) -> bool:
        """Returns True if objs1 dominates objs2 (lower is better in every dimension)."""
        return np.all(objs1 <= objs2) and np.any(objs1 < objs2)

    def update(self, new_positions: List[np.ndarray], new_genomes: List[Dict[str, Any]],
               new_infos: List[Dict[str, Any]]):
        """Filters incoming swarm entries using Pareto dominance and adds the non-dominated
        ones to the archive, evicting any existing members the new entries dominate."""
        for pos, genome, info in zip(new_positions, new_genomes, new_infos):
            objs = self._to_objectives(info)

            dominated_by_archive = False
            indices_to_remove = []
            for idx, arch_objs in enumerate(self.objectives):
                if self._dominates(arch_objs, objs):
                    dominated_by_archive = True
                    break
                if self._dominates(objs, arch_objs):
                    indices_to_remove.append(idx)

            if not dominated_by_archive:
                for idx in sorted(indices_to_remove, reverse=True):
                    self.positions.pop(idx)
                    self.genomes.pop(idx)
                    self.objectives.pop(idx)
                    self.infos.pop(idx)

                self.positions.append(pos.copy())
                self.genomes.append(copy.deepcopy(genome))
                self.objectives.append(objs)
                self.infos.append(info)

        if len(self.positions) > self.max_size:
            self._prune_archive()

    def _compute_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Creates hypercube bounds across the current objective profiles."""
        objs_matrix = np.array(self.objectives)
        min_vals = objs_matrix.min(axis=0)
        max_vals = objs_matrix.max(axis=0)

        for d in range(self.N_OBJ):
            if abs(max_vals[d] - min_vals[d]) < 1e-8:
                min_vals[d] -= 0.1
                max_vals[d] += 0.1

        steps = (max_vals - min_vals) / self.n_divisions
        return min_vals, max_vals, steps

    def _get_hypercube_indices(self) -> List[Tuple[int, ...]]:
        """Maps each archive member to a discrete grid cell (hypercube) in objective space."""
        min_vals, _, steps = self._compute_grid()
        grid_coords = []
        for objs in self.objectives:
            coords = []
            for d in range(self.N_OBJ):
                c_idx = int((objs[d] - min_vals[d]) // steps[d])
                c_idx = max(0, min(c_idx, self.n_divisions - 1))
                coords.append(c_idx)
            grid_coords.append(tuple(coords))
        return grid_coords

    def select_leader(self) -> np.ndarray:
        """Selects a gbest leader via roulette-wheel selection, inversely proportional to
        the density of the hypercube each candidate lives in (favors sparse/isolated
        regions of the Pareto front, which drives archive diversity)."""
        if not self.positions:
            return random_particle_position()

        coords_list = self._get_hypercube_indices()

        cube_counts: Dict[Tuple[int, ...], int] = {}
        for coord in coords_list:
            cube_counts[coord] = cube_counts.get(coord, 0) + 1

        cube_fitness = {coord: 10.0 / count for coord, count in cube_counts.items()}

        unique_cubes = list(cube_fitness.keys())
        fits = [cube_fitness[c] for c in unique_cubes]
        total_fit = sum(fits)

        pick = random.uniform(0, total_fit)
        current = 0.0
        selected_cube = unique_cubes[-1]
        for cube, f in zip(unique_cubes, fits):
            current += f
            if current >= pick:
                selected_cube = cube
                break

        matching_indices = [i for i, coord in enumerate(coords_list) if coord == selected_cube]
        chosen_idx = random.choice(matching_indices)
        return self.positions[chosen_idx].copy()

    def _prune_archive(self):
        """Evicts members from the most crowded hypercubes until the archive is back
        within its size limit, preserving diversity across the Pareto front."""
        while len(self.positions) > self.max_size:
            coords_list = self._get_hypercube_indices()
            cube_counts: Dict[Tuple[int, ...], int] = {}
            for coord in coords_list:
                cube_counts[coord] = cube_counts.get(coord, 0) + 1

            max_crowd = max(cube_counts.values())
            crowded_cubes = [c for c, count in cube_counts.items() if count == max_crowd]
            target_cube = random.choice(crowded_cubes)

            indices_in_cube = [i for i, coord in enumerate(coords_list) if coord == target_cube]
            kill_idx = random.choice(indices_in_cube)

            self.positions.pop(kill_idx)
            self.genomes.pop(kill_idx)
            self.objectives.pop(kill_idx)
            self.infos.pop(kill_idx)

    def get_best_by_preference(self, alpha=FITNESS_W_ACC, beta=FITNESS_W_PARAMS,
                                 gamma=FITNESS_W_FLOPS, delta=FITNESS_W_LATENCY,
                                 epsilon=FITNESS_W_MEMORY) -> Dict[str, Any]:
        """Extracts a single representative genome from the Pareto archive, using the
        SAME normalized scalar preference score as compute_population_fitness() in
        This is only used to pick one model for the rest of the pipeline
        (final training, benchmarking, MRL fine-tuning); it plays no role in the search."""
        if not self.genomes:
            return position_to_genome(random_particle_position())

        fits = compute_population_fitness(self.infos)
        best_idx = int(np.argmax(fits))
        return self.genomes[best_idx]


def _apply_mopso_mutation(pos: np.ndarray, current_iter: int, total_iters: int):
    """Adaptive exploration mutation operator: the probability of perturbing a random
    dimension decays over the run, giving the swarm broad exploration early on and
    increasingly fine-grained exploitation as it converges."""
    pm = (1.0 - (current_iter / total_iters)) ** 1.5
    if random.random() < pm:
        dim_to_mutate = random.randint(0, len(pos) - 1)
        pos[dim_to_mutate] = random_particle_position()[dim_to_mutate]


def run_mopso(train_ds, val_ds, device,
              swarm_size: int = None, generations: int = None,
              epochs_per_eval: int = None, patience: int = None,
              archive_size: int = None, n_divisions: int = None,
              w: float = None, c1: float = None, c2: float = None,
              seed: int = None, ckpt_path=None, quiet: bool = False):
    """Runs the MOPSO search. All hyperparameters default to the module-level globals but
    are overridable per-call, so the ablation study (ablation study) and hyperparameter
    sensitivity sweep (hyperparameter sensitivity sweep) can run at different/reduced budgets without mutating
    global state. Returns (best_genome, gbest_fit, history, archive) -- the archive is new
    vs. the original version and is needed for Pareto-front plotting and the proxy-training
    validity check."""
    swarm_size      = GA_SWARM_SIZE   if swarm_size      is None else swarm_size
    generations     = GA_GENERATIONS  if generations     is None else generations
    epochs_per_eval = GA_EPOCHS_PER_EVAL if epochs_per_eval is None else epochs_per_eval
    patience        = GA_PATIENCE     if patience        is None else patience
    archive_size    = GA_ARCHIVE_SIZE if archive_size    is None else archive_size
    n_divisions     = GA_N_DIVISIONS  if n_divisions     is None else n_divisions
    w  = PSO_W  if w  is None else w
    c1 = PSO_C1 if c1 is None else c1
    c2 = PSO_C2 if c2 is None else c2
    ckpt_path = GA_CKPT_PATH if ckpt_path is None else ckpt_path

    if seed is not None:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    if not quiet:
        log.info("\n" + "=" * 72)
        log.info("MOPSO MULTI-OBJECTIVE ARCHITECTURE SEARCH (Coello Coello Pareto framework)")
        log.info(f"swarm={swarm_size} | generations={generations} | epochs/eval={epochs_per_eval}")
        log.info(f"objectives (Pareto, lower=better) = [-Accuracy, Parameters, FLOPs, Inference_ms, Peak_Memory_MB]")
        log.info(f"final preference score = {FITNESS_W_ACC}*Acc - {FITNESS_W_PARAMS}*Params - {FITNESS_W_FLOPS}*FLOPs "
                  f"- {FITNESS_W_LATENCY}*Latency - {FITNESS_W_MEMORY}*Memory  (used only to pick one model "
                  f"out of the final archive)")
        log.info("=" * 72)

    archive = MOPSOArchive(max_size=archive_size, n_divisions=n_divisions)

    positions   = [random_particle_position() for _ in range(swarm_size)]
    velocities  = [np.zeros(PSO_N_DIMS) for _ in range(swarm_size)]
    v_max       = np.array(GA_GENE_RANGES, dtype=float) * 0.5


    pbest_pos   = [p.copy() for p in positions]
    pbest_objs  = [np.array([np.inf] * MOPSOArchive.N_OBJ) for _ in range(swarm_size)]

    history = []
    gbest_fit, gbest_genome = -np.inf, None

    for gen in range(1, generations + 1):
        t_gen = time.time()
        current_genomes = [position_to_genome(pos) for pos in positions]
        current_infos = []

        for idx, genome in enumerate(current_genomes):
            _, info = evaluate_individual_multiobjective(genome, train_ds, val_ds, device,
                                                          epochs_per_eval=epochs_per_eval,
                                                          patience=patience)
            current_infos.append(info)
            if not quiet:
                log.info(f"  [{idx+1:02d}/{swarm_size}] acc={info['accuracy']:.2f}% "
                          f"params={info['n_params']:,} flops={info['flops']:,} "
                          f"infer={info['inference_ms']:.3f}ms mem={info['peak_memory_mb']:.1f}MB")


        archive.update(positions, current_genomes, current_infos)
        if not quiet:
            log.info(f"  >> Pareto Archive Size: {len(archive.positions)} / {archive.max_size}")


        fits = compute_population_fitness(current_infos)
        for idx in range(swarm_size):
            if fits[idx] > gbest_fit:
                gbest_fit = fits[idx]
                gbest_genome = current_genomes[idx]
                _save_genome(gbest_genome, gbest_fit, ckpt_path)
                if not quiet:
                    log.info(f"  \u2605 New global best preference fitness = {gbest_fit:.5f}")


        for idx in range(swarm_size):
            c_objs = archive._to_objectives(current_infos[idx])
            if np.all(c_objs <= pbest_objs[idx]) and np.any(c_objs < pbest_objs[idx]):
                pbest_objs[idx] = c_objs.copy()
                pbest_pos[idx]  = positions[idx].copy()
            elif not (np.all(pbest_objs[idx] <= c_objs) and np.any(pbest_objs[idx] < c_objs)):

                if random.random() < 0.5:
                    pbest_objs[idx] = c_objs.copy()
                    pbest_pos[idx]  = positions[idx].copy()


        for idx in range(swarm_size):
            gbest_pos = archive.select_leader()

            r1, r2 = random.random(), random.random()
            velocities[idx] = (w * velocities[idx]
                               + c1 * r1 * (pbest_pos[idx] - positions[idx])
                               + c2 * r2 * (gbest_pos - positions[idx]))
            velocities[idx] = np.clip(velocities[idx], -v_max, v_max)
            positions[idx] = positions[idx] + velocities[idx]

            _apply_mopso_mutation(positions[idx], gen, generations)

        elapsed_gen = time.time() - t_gen
        history.append({"iteration": gen, "best_fitness": gbest_fit,
                          "archive_count": len(archive.positions), "elapsed_s": elapsed_gen})
        if not quiet:
            log.info(f"  MOPSO gen {gen}/{generations} done | gbest={gbest_fit:.5f} | "
                      f"archive={len(archive.positions)} | elapsed={elapsed_gen:.0f}s")

    if not quiet:
        log.info("\n" + "=" * 72)
        log.info(f"MOPSO COMPLETE -- best preference fitness = {gbest_fit:.5f} | "
                  f"final archive size = {len(archive.positions)}")
        log.info("=" * 72)

    best_genome = archive.get_best_by_preference(alpha=FITNESS_W_ACC, beta=FITNESS_W_PARAMS,
                                                   gamma=FITNESS_W_FLOPS, delta=FITNESS_W_LATENCY,
                                                   epsilon=FITNESS_W_MEMORY)
    if best_genome is None:
        best_genome = gbest_genome

    return best_genome, gbest_fit, history, archive
