from ...config import *
from ...seeding import set_all_seeds
from ..encoding import chromosome_to_genome, ga_mutate, ga_tournament_select, ga_uniform_crossover, random_chromosome
from ..objectives import FITNESS_W_ACC, FITNESS_W_FLOPS, FITNESS_W_LATENCY, FITNESS_W_MEMORY, FITNESS_W_PARAMS, compute_population_fitness, evaluate_individual_multiobjective

def run_ga(train_ds, val_ds, device, pop_size: int = None, generations: int = None,
           epochs_per_eval: int = None, patience: int = None,
           crossover_rate: float = None, mut_prob: float = None, elitism: int = None,
           tournament_k: int = None, seed: int = None, ckpt_path=None, quiet: bool = False):
    """Runs the genetic algorithm search. All hyperparameters default to the shared module-
    level globals (same values used by PSO/MOPSO) but are overridable per-call. Returns
    (best_genome, gbest_fit, history) -- same return shape as run_pso so downstream code
    (final training / benchmarking / significance tests) is symmetric."""
    pop_size        = GA_POP_SIZE        if pop_size        is None else pop_size
    generations     = GA_GENERATIONS     if generations     is None else generations
    epochs_per_eval = GA_EPOCHS_PER_EVAL if epochs_per_eval is None else epochs_per_eval
    patience        = GA_PATIENCE        if patience        is None else patience
    crossover_rate  = GA_CROSSOVER_RATE  if crossover_rate  is None else crossover_rate
    mut_prob        = GA_MUT_PROB        if mut_prob        is None else mut_prob
    elitism         = GA_ELITISM         if elitism         is None else elitism
    tournament_k    = GA_TOURNAMENT_K    if tournament_k    is None else tournament_k
    ckpt_path       = Path("best_ga_genome.json") if ckpt_path is None else ckpt_path

    if seed is not None:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    if not quiet:
        log.info("\n" + "=" * 72)
        log.info("GENETIC ALGORITHM ARCHITECTURE SEARCH (discrete/categorical chromosome)")
        log.info(f"pop={pop_size} | generations={generations} | epochs/eval={epochs_per_eval} | "
                 f"crossover_rate={crossover_rate} | mut_prob={mut_prob} | elitism={elitism}")
        log.info(f"fitness = {FITNESS_W_ACC}*Acc - {FITNESS_W_PARAMS}*Params - {FITNESS_W_FLOPS}*FLOPs "
                 f"- {FITNESS_W_LATENCY}*Latency - {FITNESS_W_MEMORY}*Memory  (identical to PSO/MOPSO)")
        log.info("=" * 72)

    population = [random_chromosome() for _ in range(pop_size)]

    history = []
    gbest_fit, gbest_genome = -np.inf, None

    for gen in range(1, generations + 1):
        t_gen = time.time()
        current_genomes = [chromosome_to_genome(chrom) for chrom in population]
        current_infos = []

        for idx, genome in enumerate(current_genomes):
            _, info = evaluate_individual_multiobjective(genome, train_ds, val_ds, device,
                                                          epochs_per_eval=epochs_per_eval,
                                                          patience=patience)
            current_infos.append(info)
            if not quiet:
                log.info(f"  [{idx+1:02d}/{pop_size}] acc={info['accuracy']:.2f}% "
                         f"params={info['n_params']:,} flops={info['flops']:,} "
                         f"infer={info['inference_ms']:.3f}ms mem={info['peak_memory_mb']:.1f}MB")

        fits = compute_population_fitness(current_infos)

        for idx in range(pop_size):
            if fits[idx] > gbest_fit:
                gbest_fit = fits[idx]
                gbest_genome = current_genomes[idx]
                with open(ckpt_path, "w") as fh:
                    json.dump({"genome": gbest_genome, "fitness": gbest_fit}, fh, indent=2)
                if not quiet:
                    log.info(f"   New GA global best fitness = {gbest_fit:.5f}")


        elite_order = sorted(range(pop_size), key=lambda i: fits[i], reverse=True)
        elites = [population[i].copy() for i in elite_order[:elitism]]


        next_population = list(elites)
        while len(next_population) < pop_size:
            parent_a = ga_tournament_select(population, fits, k=tournament_k)
            parent_b = ga_tournament_select(population, fits, k=tournament_k)
            child_a, child_b = ga_uniform_crossover(parent_a, parent_b, crossover_rate=crossover_rate)
            child_a = ga_mutate(child_a, mut_prob=mut_prob)
            next_population.append(child_a)
            if len(next_population) < pop_size:
                child_b = ga_mutate(child_b, mut_prob=mut_prob)
                next_population.append(child_b)

        population = next_population[:pop_size]

        elapsed_gen = time.time() - t_gen
        history.append({"iteration": gen, "best_fitness": gbest_fit, "elapsed_s": elapsed_gen})
        if not quiet:
            log.info(f"  GA gen {gen}/{generations} done | gbest={gbest_fit:.5f} | elapsed={elapsed_gen:.0f}s")

    if not quiet:
        log.info("\n" + "=" * 72)
        log.info(f"GA COMPLETE -- best fitness = {gbest_fit:.5f}")
        log.info("=" * 72)

    return gbest_genome, gbest_fit, history
