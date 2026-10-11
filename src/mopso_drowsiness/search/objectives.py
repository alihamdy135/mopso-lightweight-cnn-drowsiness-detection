from ..config import *
from ..data.mrl import EyeRegionDataset
from ..models.dynamic_cnn import SingleEyeRegionCNN
from ..training.engine import _evaluate, _run_train_epoch, count_flops, full_benchmark, measure_inference_speed, measure_peak_memory_mb, train_on_primary

FITNESS_W_ACC     = 0.50
FITNESS_W_PARAMS  = 0.15
FITNESS_W_FLOPS   = 0.15
FITNESS_W_LATENCY = 0.10
FITNESS_W_MEMORY  = 0.10


def _minmax_norm(values: List[float]) -> List[float]:
    """Normalizes a list of raw values to [0, 1]. Falls back to all-0.5 if no spread."""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def evaluate_individual_multiobjective(genome: Dict, train_ds, val_ds, device,
                                        epochs_per_eval: int = None,
                                        patience: int = None) -> Tuple[float, Dict]:
    """Trains one candidate architecture (decoded from a GA individual/chromosome) for up to
    `epochs_per_eval` epochs with early stopping, then measures all 5 raw objective values:
    accuracy, parameter count, FLOPs, inference latency, and peak memory. Normalization
    across the population happens later in compute_population_fitness().
    `epochs_per_eval`/`patience` now default to the global GA_EPOCHS_PER_EVAL/GA_PATIENCE but
    are overridable per-call -- needed so the ablation study and hyperparameter sensitivity
    sweep (Parts 10B/10C) can run at a reduced diagnostic budget without touching globals."""
    epochs_per_eval = GA_EPOCHS_PER_EVAL if epochs_per_eval is None else epochs_per_eval
    patience = GA_PATIENCE if patience is None else patience

    try:
        model = SingleEyeRegionCNN(genome, num_classes=2).to(device)
    except Exception as exc:
        log.warning(f"  Model build failed: {exc}")
        return -1.0, {"accuracy": 0.0, "n_params": np.inf, "flops": np.inf,
                      "inference_ms": np.inf, "peak_memory_mb": np.inf,
                      "depth": genome.get("depth", 0)}

    n_params  = model.param_count
    criterion = nn.CrossEntropyLoss()
    train_ldr = DataLoader(train_ds, batch_size=GA_BATCH, shuffle=True, pin_memory=True)
    val_ldr   = DataLoader(val_ds, batch_size=GA_BATCH, shuffle=False, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=GA_LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_per_eval)

    best_acc, no_imp = 0.0, 0
    for ep in range(1, epochs_per_eval + 1):
        _run_train_epoch(model, train_ldr, criterion, optimizer, device)
        scheduler.step()
        m = _evaluate(model, val_ldr, criterion, device)
        acc = m.get("accuracy", 0.0)
        if acc > best_acc + 1e-5:
            best_acc, no_imp = acc, 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break


    flops          = count_flops(model)
    ms_per_img, _  = measure_inference_speed(model, device, n_runs=50)
    peak_memory_mb = measure_peak_memory_mb(model, device)

    info = {
        "accuracy": best_acc, "n_params": n_params, "flops": flops,
        "inference_ms": ms_per_img, "peak_memory_mb": peak_memory_mb,
        "depth": genome["depth"],
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_acc, info


def compute_population_fitness(pop_infos: List[Dict]) -> List[float]:
    """Min-max normalizes each objective across the current population, then combines them
    into a single scalar fitness per individual:
        maximize : accuracy
        minimize : parameter count, FLOPs, inference time, peak memory
    """
    accs    = [info["accuracy"]       for info in pop_infos]
    params  = [info["n_params"]       for info in pop_infos]
    flops   = [info["flops"]          for info in pop_infos]
    infer_t = [info["inference_ms"]   for info in pop_infos]
    memory  = [info["peak_memory_mb"] for info in pop_infos]

    acc_n    = _minmax_norm(accs)
    params_n = _minmax_norm(params)
    flops_n  = _minmax_norm(flops)
    infer_n  = _minmax_norm(infer_t)
    memory_n = _minmax_norm(memory)

    fitnesses = []
    for a, p, fl, t, mem in zip(acc_n, params_n, flops_n, infer_n, memory_n):
        fit = (FITNESS_W_ACC * a
               - FITNESS_W_PARAMS  * p
               - FITNESS_W_FLOPS   * fl
               - FITNESS_W_LATENCY * t
               - FITNESS_W_MEMORY  * mem)
        fitnesses.append(fit)
    return fitnesses
