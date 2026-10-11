from ...config import *
from ...seeding import set_all_seeds
from ..encoding import chromosome_to_genome, ga_mutate, ga_uniform_crossover, random_chromosome
from ..objectives import compute_population_fitness, evaluate_individual_multiobjective
from ..pareto import _objectives_from_info, crowding_distance, fast_non_dominated_sort, nsga2_tournament

def run_nsga2(train_ds, val_ds, device, pop_size: int = None, generations: int = None,
              epochs_per_eval: int = None, patience: int = None,
              crossover_rate: float = None, mut_prob: float = None,
              tournament_k: int = 2, seed: int = None, ckpt_path=None, quiet: bool = False):
    """Runs the NSGA-II search. Returns (best_genome, gbest_fit, history, final_front) --
    final_front is a dict with the last generation's non-dominated front's genomes/infos,
    mirroring the archive MOPSO returns, for Pareto-front plotting/reporting."""
    pop_size        = GA_POP_SIZE        if pop_size        is None else pop_size
    generations     = GA_GENERATIONS     if generations     is None else generations
    epochs_per_eval = GA_EPOCHS_PER_EVAL if epochs_per_eval is None else epochs_per_eval
    patience        = GA_PATIENCE        if patience        is None else patience
    crossover_rate  = GA_CROSSOVER_RATE  if crossover_rate  is None else crossover_rate
    mut_prob        = GA_MUT_PROB        if mut_prob        is None else mut_prob
    ckpt_path       = Path("best_nsga2_genome.json") if ckpt_path is None else ckpt_path

    if seed is not None:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    if not quiet:
        log.info("\n" + "=" * 72)
        log.info("NSGA-II MULTI-OBJECTIVE ARCHITECTURE SEARCH (Deb et al., 2002)")
        log.info(f"pop={pop_size} | generations={generations} | epochs/eval={epochs_per_eval} | "
                 f"crossover_rate={crossover_rate} | mut_prob={mut_prob}")
        log.info("objectives (Pareto, lower=better) = [-Accuracy, Parameters, FLOPs, Inference_ms, Peak_Memory_MB]")
        log.info("=" * 72)


    population = [random_chromosome() for _ in range(pop_size)]
    genomes = [chromosome_to_genome(c) for c in population]
    infos = []
    for idx, genome in enumerate(genomes):
        _, info = evaluate_individual_multiobjective(genome, train_ds, val_ds, device,
                                                      epochs_per_eval=epochs_per_eval,
                                                      patience=patience)
        infos.append(info)
        if not quiet:
            log.info(f"  [{idx+1:02d}/{pop_size}] acc={info['accuracy']:.2f}% "
                     f"params={info['n_params']:,} flops={info['flops']:,} "
                     f"infer={info['inference_ms']:.3f}ms mem={info['peak_memory_mb']:.1f}MB")

    history = []
    gbest_fit, gbest_genome = -np.inf, None
    final_front_infos, final_front_genomes = [], []

    for gen in range(1, generations + 1):
        t_gen = time.time()

        objs = [_objectives_from_info(info) for info in infos]
        fronts = fast_non_dominated_sort(objs)
        ranks, crowd = {}, {}
        for r, front in enumerate(fronts):
            cd = crowding_distance(front, objs)
            for idx in front:
                ranks[idx] = r
                crowd[idx] = cd[idx]


        fits = compute_population_fitness(infos)
        for idx in range(len(population)):
            if fits[idx] > gbest_fit:
                gbest_fit = fits[idx]
                gbest_genome = genomes[idx]
                with open(ckpt_path, "w") as fh:
                    json.dump({"genome": gbest_genome, "fitness": gbest_fit}, fh, indent=2)
                if not quiet:
                    log.info(f"  \u2605 New NSGA-II global best preference fitness = {gbest_fit:.5f}")

        final_front_infos   = [infos[i]   for i in fronts[0]]
        final_front_genomes = [genomes[i] for i in fronts[0]]


        all_idx = list(range(len(population)))
        offspring_chroms = []
        while len(offspring_chroms) < pop_size:
            a = nsga2_tournament(all_idx, ranks, crowd, k=tournament_k)
            b = nsga2_tournament(all_idx, ranks, crowd, k=tournament_k)
            child_a, child_b = ga_uniform_crossover(population[a], population[b],
                                                     crossover_rate=crossover_rate)
            child_a = ga_mutate(child_a, mut_prob=mut_prob)
            offspring_chroms.append(child_a)
            if len(offspring_chroms) < pop_size:
                child_b = ga_mutate(child_b, mut_prob=mut_prob)
                offspring_chroms.append(child_b)

        offspring_genomes = [chromosome_to_genome(c) for c in offspring_chroms]
        offspring_infos = []
        for idx, genome in enumerate(offspring_genomes):
            _, info = evaluate_individual_multiobjective(genome, train_ds, val_ds, device,
                                                          epochs_per_eval=epochs_per_eval,
                                                          patience=patience)
            offspring_infos.append(info)
            if not quiet:
                log.info(f"  [gen{gen} offspring {idx+1:02d}/{pop_size}] acc={info['accuracy']:.2f}% "
                         f"params={info['n_params']:,} flops={info['flops']:,}")


        combined_chroms  = population + offspring_chroms
        combined_genomes = genomes + offspring_genomes
        combined_infos   = infos + offspring_infos
        combined_objs    = objs + [_objectives_from_info(info) for info in offspring_infos]

        combined_fronts = fast_non_dominated_sort(combined_objs)
        next_idx = []
        for front in combined_fronts:
            if len(next_idx) + len(front) <= pop_size:
                next_idx.extend(front)
            else:
                remaining = pop_size - len(next_idx)
                cd = crowding_distance(front, combined_objs)
                front_sorted = sorted(front, key=lambda i: -cd[i])
                next_idx.extend(front_sorted[:remaining])
                break

        population = [combined_chroms[i]  for i in next_idx]
        genomes    = [combined_genomes[i] for i in next_idx]
        infos      = [combined_infos[i]   for i in next_idx]

        elapsed_gen = time.time() - t_gen
        history.append({"iteration": gen, "best_fitness": gbest_fit, "elapsed_s": elapsed_gen})
        if not quiet:
            log.info(f"  NSGA-II gen {gen}/{generations} done | gbest={gbest_fit:.5f} | "
                     f"front0={len(final_front_genomes)} | elapsed={elapsed_gen:.0f}s")

    if not quiet:
        log.info("\n" + "=" * 72)
        log.info(f"NSGA-II COMPLETE -- best preference fitness = {gbest_fit:.5f} | "
                 f"final Pareto front size = {len(final_front_genomes)}")
        log.info("=" * 72)


    if final_front_infos:
        front_fits = compute_population_fitness(final_front_infos)
        best_genome = final_front_genomes[int(np.argmax(front_fits))]
    else:
        best_genome = gbest_genome

    return best_genome, gbest_fit, history, {"genomes": final_front_genomes, "infos": final_front_infos}
