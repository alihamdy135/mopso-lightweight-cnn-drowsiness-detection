from ...config import *
from ...seeding import set_all_seeds
from ..encoding import GA_GENE_RANGES, PSO_N_DIMS, position_to_genome, random_particle_position
from ..objectives import FITNESS_W_ACC, FITNESS_W_FLOPS, FITNESS_W_LATENCY, FITNESS_W_MEMORY, FITNESS_W_PARAMS, compute_population_fitness, evaluate_individual_multiobjective

def run_pso(train_ds, val_ds, device, swarm_size: int = None, generations: int = None,
            epochs_per_eval: int = None, patience: int = None,
            w: float = None, c1: float = None, c2: float = None,
            seed: int = None, ckpt_path=None, quiet: bool = False):
    """Runs single-objective PSO search. Returns (best_genome, gbest_fit, history) -- same
    shape as run_ga so downstream code (final training / benchmarking / significance
    tests) is symmetric across the two algorithms being statistically compared."""
    swarm_size      = GA_POP_SIZE         if swarm_size      is None else swarm_size
    generations     = GA_GENERATIONS      if generations     is None else generations
    epochs_per_eval = GA_EPOCHS_PER_EVAL  if epochs_per_eval is None else epochs_per_eval
    patience        = GA_PATIENCE         if patience        is None else patience
    w  = PSO_W  if w  is None else w
    c1 = PSO_C1 if c1 is None else c1
    c2 = PSO_C2 if c2 is None else c2
    ckpt_path = Path("best_pso_genome.json") if ckpt_path is None else ckpt_path

    if seed is not None:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    if not quiet:
        log.info("\n" + "=" * 72)
        log.info("SINGLE-OBJECTIVE PSO ARCHITECTURE SEARCH")
        log.info(f"swarm={swarm_size} | generations={generations} | epochs/eval={epochs_per_eval}")
        log.info(f"fitness = {FITNESS_W_ACC}*Acc - {FITNESS_W_PARAMS}*Params - {FITNESS_W_FLOPS}*FLOPs "
                 f"- {FITNESS_W_LATENCY}*Latency - {FITNESS_W_MEMORY}*Memory  (identical to GA/MOPSO)")
        log.info("=" * 72)

    positions  = [random_particle_position() for _ in range(swarm_size)]
    velocities = [np.zeros(PSO_N_DIMS) for _ in range(swarm_size)]
    v_max      = np.array(GA_GENE_RANGES, dtype=float) * 0.5
    pbest_pos  = [p.copy() for p in positions]
    pbest_fit  = [-np.inf] * swarm_size

    gbest_fit, gbest_pos, gbest_genome = -np.inf, positions[0].copy(), None
    history = []

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

        fits = compute_population_fitness(current_infos)

        for idx in range(swarm_size):
            if fits[idx] > pbest_fit[idx]:
                pbest_fit[idx] = fits[idx]
                pbest_pos[idx] = positions[idx].copy()
            if fits[idx] > gbest_fit:
                gbest_fit = fits[idx]
                gbest_pos = positions[idx].copy()
                gbest_genome = current_genomes[idx]
                with open(ckpt_path, "w") as fh:
                    json.dump({"genome": gbest_genome, "fitness": gbest_fit}, fh, indent=2)
                if not quiet:
                    log.info(f"   New PSO global best fitness = {gbest_fit:.5f}")

        for idx in range(swarm_size):
            r1, r2 = random.random(), random.random()
            velocities[idx] = (w * velocities[idx]
                               + c1 * r1 * (pbest_pos[idx] - positions[idx])
                               + c2 * r2 * (gbest_pos - positions[idx]))
            velocities[idx] = np.clip(velocities[idx], -v_max, v_max)
            positions[idx] = positions[idx] + velocities[idx]

        elapsed_gen = time.time() - t_gen
        history.append({"iteration": gen, "best_fitness": gbest_fit, "elapsed_s": elapsed_gen})
        if not quiet:
            log.info(f"  PSO gen {gen}/{generations} done | gbest={gbest_fit:.5f} | elapsed={elapsed_gen:.0f}s")

    if not quiet:
        log.info("\n" + "=" * 72)
        log.info(f"PSO COMPLETE -- best fitness = {gbest_fit:.5f}")
        log.info("=" * 72)

    return gbest_genome, gbest_fit, history
