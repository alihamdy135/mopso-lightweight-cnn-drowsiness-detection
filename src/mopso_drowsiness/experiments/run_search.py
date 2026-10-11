from ..config import *
from ..data.cew import get_cew_finetune_test_split
from ..data.mrl import EyeRegionDataset, get_mrl_search_proxy, get_mrl_search_split
from ..evaluation.benchmark import plot_convergence, train_finetune_benchmark
from ..models.dynamic_cnn import SingleEyeRegionCNN
from ..search.algorithms.ga import run_ga
from ..search.algorithms.mopso import run_mopso
from ..search.algorithms.nsga2 import run_nsga2
from ..search.algorithms.pso import run_pso
from .seed_runner import ga_vs_pso_significance, run_algorithm_n_times


def run_search_experiment(include_primary_benchmarks: bool = True,
                          include_seed_evaluation: bool = True) -> dict:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, search_val_ds = get_mrl_search_split()
    search_train_ds = get_mrl_search_proxy(train_ds, frac=SEARCH_PROXY_FRAC, seed=SEARCH_SEED)

    primary_searches = {}
    ga_best_genome, ga_best_fit, ga_history = run_ga(search_train_ds, search_val_ds, DEVICE, seed=SEARCH_SEED, quiet=False)
    pso_best_genome, pso_best_fit, pso_history = run_pso(search_train_ds, search_val_ds, DEVICE, seed=SEARCH_SEED, quiet=False)
    mopso_best_genome, mopso_best_fit, mopso_history, mopso_archive = run_mopso(search_train_ds, search_val_ds, DEVICE, seed=SEARCH_SEED, quiet=False)
    nsga2_best_genome, nsga2_best_fit, nsga2_history, nsga2_front = run_nsga2(search_train_ds, search_val_ds, DEVICE, seed=NSGA2_SEARCH_SEED, quiet=False)

    primary_searches = {
        "GA": {"genome": ga_best_genome, "fitness": ga_best_fit, "history": ga_history},
        "PSO": {"genome": pso_best_genome, "fitness": pso_best_fit, "history": pso_history},
        "MOPSO": {"genome": mopso_best_genome, "fitness": mopso_best_fit, "history": mopso_history, "archive": mopso_archive},
        "NSGA-II": {"genome": nsga2_best_genome, "fitness": nsga2_best_fit, "history": nsga2_history, "front": nsga2_front},
    }
    log.info(f"\n[Search complete] GA best fitness={ga_best_fit:.5f} | PSO best fitness={pso_best_fit:.5f} | MOPSO best fitness={mopso_best_fit:.5f}")
    log.info(f"\n[Search complete] NSGA-II best fitness={nsga2_best_fit:.5f}")
    plot_convergence({name: record["history"] for name, record in primary_searches.items()})

    primary_results = []
    primary_seed_results = {}
    seed_results_df = pd.DataFrame()
    seed_summary_df = pd.DataFrame()
    significance_df = pd.DataFrame()

    if include_primary_benchmarks or include_seed_evaluation:
        mrl_test_ds = EyeRegionDataset(MRL_TEST_ROOT, {key: [key] for key in MRL_CLASS_MAP}, MRL_CLASS_MAP)
        mrl_test_loader = DataLoader(mrl_test_ds, batch_size=GA_BATCH, shuffle=False, pin_memory=True)

    if include_primary_benchmarks:
        cew_finetune_ds, cew_test_ds = get_cew_finetune_test_split()
        cew_finetune_loader = DataLoader(cew_finetune_ds, batch_size=GA_BATCH, shuffle=True)
        cew_test_loader = DataLoader(cew_test_ds, batch_size=GA_BATCH, shuffle=False)
        for method_name, search_result in primary_searches.items():
            model, mrl_bench, cew_bench = train_finetune_benchmark(
                search_result["genome"], method_name, train_ds, search_val_ds,
                mrl_test_loader, cew_finetune_loader, cew_test_loader, DEVICE)
            primary_results.extend([mrl_bench, cew_bench])
            primary_seed = NSGA2_SEARCH_SEED if method_name == "NSGA-II" else SEARCH_SEED
            primary_seed_results[method_name] = {"Seed": primary_seed, "Accuracy": mrl_bench["Accuracy"]}
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        primary_results_df = pd.DataFrame(primary_results)
        primary_results_df.to_csv(RESULTS_DIR / "primary_benchmarks.csv", index=False)
        print(primary_results_df.to_string(index=False))

    method_records = {}
    if include_seed_evaluation:
        method_specs = [
            ("GA", run_ga, GA_SEED_START),
            ("PSO", run_pso, GA_SEED_START),
            ("MOPSO", run_mopso, GA_SEED_START),
            ("NSGA-II", run_nsga2, NSGA2_SEED_START),
        ]
        for method_name, search_fn, base_seed in method_specs:
            method_records[method_name] = run_algorithm_n_times(
                search_fn, train_ds, search_val_ds, mrl_test_loader, DEVICE,
                n_runs=N_STAT_RUNS, base_seed=base_seed, tag=method_name,
                search_train_ds=search_train_ds,
                reuse_record=primary_seed_results.get(method_name))

        seed_results_df = pd.DataFrame([
            {"Method": method_name, **record}
            for method_name, records in method_records.items()
            for record in records
        ])
        seed_summary_df = seed_results_df.groupby("Method")["Accuracy"].agg(
            Runs="count", Mean_MRL_Test_Accuracy="mean", Std_MRL_Test_Accuracy="std"
        ).reset_index()
        seed_results_df.to_csv(RESULTS_DIR / "seed_results.csv", index=False)
        seed_summary_df.to_csv(RESULTS_DIR / "seed_summary.csv", index=False)
        print(seed_results_df.to_string(index=False))
        print(seed_summary_df.to_string(index=False))
        significance_df = ga_vs_pso_significance(
            [record["Accuracy"] for record in method_records["GA"]],
            [record["Accuracy"] for record in method_records["PSO"]])
        significance_df.to_csv(RESULTS_DIR / "ga_vs_pso_significance.csv", index=False)
        print(significance_df.to_string(index=False))

    return {
        "primary_searches": primary_searches,
        "primary_results": pd.DataFrame(primary_results),
        "method_records": method_records,
        "seed_results": seed_results_df,
        "seed_summary": seed_summary_df,
        "significance": significance_df,
    }
