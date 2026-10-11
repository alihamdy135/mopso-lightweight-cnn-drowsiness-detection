from ..config import *
from ..data.mrl import EyeRegionDataset
from ..models.dynamic_cnn import SingleEyeRegionCNN
from ..seeding import seed_schedule
from ..training.engine import full_benchmark, train_on_primary

def run_algorithm_n_times(search_fn, train_ds, search_val_ds, mrl_test_loader, device,
                           n_runs: int = N_STAT_RUNS, base_seed: int = 106, tag: str = "algo",
                           search_train_ds=None, reuse_record=None):
    """Runs a selected search function for n_runs distinct seeds and trains the
    best genome from each run on MRL, and benchmarks it on the held-out MRL test set,
    collecting each seed and its test accuracy for significance testing. The primary
    benchmark result is reused for its matching seed so each method has exactly n_runs unique seeds.
    `search_train_ds`: the proxy subset (SEARCH_PROXY_FRAC of train_ds, see
    `get_mrl_search_proxy` defined above) that `search_fn` actually trains candidates on during
    the search loop -- defaults to the full `train_ds` if not supplied. Final training of
    each run's selected best genome always uses the FULL `train_ds`, matching the full-data training protocol."""
    search_train_ds = train_ds if search_train_ds is None else search_train_ds
    accs = []
    for run, seed in enumerate(seed_schedule(base_seed, n_runs)):
        if reuse_record is not None and seed == reuse_record["Seed"]:
            accs.append(dict(reuse_record))
            log.info(f"  [{tag}] reusing primary benchmark for seed={seed}.")
            continue
        search_result = search_fn(search_train_ds, search_val_ds, device, seed=seed, quiet=True)
        best_genome = search_result[0]

        model = SingleEyeRegionCNN(best_genome, num_classes=2).to(device)
        train_loader = DataLoader(train_ds, batch_size=GA_BATCH, shuffle=True, pin_memory=True)
        val_loader   = DataLoader(search_val_ds, batch_size=GA_BATCH, shuffle=False, pin_memory=True)
        model = train_on_primary(model, train_loader, val_loader, device,
                                  epochs=PRIMARY_EPOCHS, lr=GA_LR, tag=f"{tag}_sigrun{run}")

        bench = full_benchmark(model, mrl_test_loader, device, tag=f"{tag}_run{run}")
        accs.append({"Seed": seed, "Accuracy": bench["Accuracy"]})
        log.info(f"  [{tag}] significance run {run+1}/{n_runs} (seed={seed}): "
                 f"MRL test_acc={bench['Accuracy']:.2f}%")

    return accs

def ga_vs_pso_significance(ga_accs: list, pso_accs: list) -> pd.DataFrame:
    """Paired t-test + Wilcoxon signed-rank test comparing GA vs PSO final MRL test
    accuracies across the paired-seed runs collected above."""
    ga_arr, pso_arr = np.array(ga_accs), np.array(pso_accs)
    n = min(len(ga_arr), len(pso_arr))
    t_stat, t_p = ttest_rel(ga_arr[:n], pso_arr[:n])
    try:
        w_stat, w_p = wilcoxon(ga_arr[:n], pso_arr[:n])
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    return pd.DataFrame([{
        "Comparison": "GA vs PSO",
        "GA mean acc": ga_arr[:n].mean(), "PSO mean acc": pso_arr[:n].mean(),
        "Paired t-stat": t_stat, "Paired t p-value": t_p,
        "Wilcoxon stat": w_stat, "Wilcoxon p-value": w_p,
        "Significant (p<0.05)": bool(t_p < 0.05),
    }]).round(4)
