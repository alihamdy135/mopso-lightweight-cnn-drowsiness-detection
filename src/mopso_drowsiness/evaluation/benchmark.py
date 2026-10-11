from ..config import *
from ..data.cew import get_cew_finetune_test_split
from ..models.dynamic_cnn import SingleEyeRegionCNN
from ..training.engine import finetune_on_secondary, full_benchmark, train_on_primary

def plot_convergence(histories: Dict[str, list], savepath="./paper_figures/convergence.png"):
    plt.figure(figsize=(7, 5))
    for label, hist in histories.items():
        gens = [h["iteration"] for h in hist]
        fits = [h["best_fitness"] for h in hist]
        plt.plot(gens, fits, marker="o", label=label)
    plt.xlabel("Generation")
    plt.ylabel("Best preference fitness")
    plt.title("Search Convergence: GA, PSO, MOPSO, and NSGA-II (matched budget)")
    plt.legend()
    plt.grid(alpha=0.3)
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.show()

def train_finetune_benchmark(genome: Dict, tag: str, train_ds, search_val_ds,
                              mrl_test_loader, cew_finetune_loader, cew_test_loader, device):
    """Runs the shared final-model protocol for one algorithm's best genome:
      1) build model from genome
      2) train on MRL (train_ds/search_val_ds for early best-checkpoint selection)
      3) benchmark on held-out MRL test set
      4) fine-tune on CEW (deduplicated)
      5) benchmark on held-out CEW test set
    Returns (model, mrl_bench, cew_bench). Identical for GA, PSO, MOPSO, and NSGA-II so the
    reported numbers are directly comparable.
    """
    model = SingleEyeRegionCNN(genome, num_classes=2).to(device)

    train_loader = DataLoader(train_ds, batch_size=GA_BATCH, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(search_val_ds, batch_size=GA_BATCH, shuffle=False, pin_memory=True)

    model = train_on_primary(model, train_loader, val_loader, device,
                              epochs=PRIMARY_EPOCHS, lr=GA_LR, tag=f"{tag}_MRL")

    mrl_bench = full_benchmark(model, mrl_test_loader, device, tag=f"{tag} (MRL test)")

    model = finetune_on_secondary(model, cew_finetune_loader, device,
                                   epochs=FT_EPOCHS, tag=f"{tag}_CEW")

    cew_bench = full_benchmark(model, cew_test_loader, device, tag=f"{tag} (CEW test)")

    return model, mrl_bench, cew_bench
