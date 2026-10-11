# MOPSO Lightweight CNN for Driver Drowsiness Detection

Research code organized from the project notebook for the manuscript *Multi-Objective Particle Swarm Optimization for Lightweight CNN Architecture Search in Real-Time Driver Drowsiness Detection* by Anoureen Hasan, Ali Hamdi, Ahmed Gomaa, and Mohamed Issa.

The repository separates the optimizer-search pipeline from the transfer-learning baselines. It does not include datasets, pretrained checkpoints, or experiment outputs. The manuscript source identifies itself as a preprint submitted to Elsevier; a final venue and DOI are not specified here.

## Research question and scope

The manuscript studies whether multi-objective search can identify compact CNN architectures for eye-image classification while considering predictive performance and deployment-oriented costs. The released code covers the four search methods included in the merged notebook—GA, PSO, MOPSO, and NSGA-II—and six transfer-learning model variants. It does not add the MOOA method.

The code and defaults are extracted from the merged research notebook. This repository snapshot has not been used to rerun training or evaluation, and no result tables are bundled. Any values in the manuscript remain manuscript-reported until linked to a verified run and artifact.

## Methods and defaults

- Search methods: GA, PSO, MOPSO, and NSGA-II.
- Candidate search budget in the notebook defaults: population/swarm size 6, 6 generations, and 8 training epochs per candidate.
- Repeated search runs: 10 seeds per method. GA, PSO, and MOPSO use seeds 100–109; NSGA-II uses 109–118. The primary benchmark seed is reused when it matches the corresponding repeated-run schedule.
- The primary MRL training schedule uses 40 epochs; CEW fine-tuning uses 10 epochs.
- Transfer baselines: `MobileNetV2_1dense`, `MobileNetV2_2dense`, `ResNet50_1dense`, `ResNet50_2dense`, `VGG16_1dense`, and `VGG16_2dense`.

Runtime defaults are collected in `configs/paper_default.yaml`. Dataset and output locations can be overridden through environment variables; no data is downloaded by the code.

## Notebook-to-module map

| Source notebook cells | Project files | Responsibility |
|---|---|---|
| 0–1, 22, 24 | `seeding.py`, `config.py`, `configs/paper_default.yaml` | Deterministic seed utility and notebook-derived runtime/search settings |
| 2 | `data/mrl.py`, `data/cew.py`, `data/splits.py` | Dataset readers, split helpers, and duplicate-image handling |
| 3–5 | `models/dynamic_cnn.py`, `training/engine.py` | Search CNN, training loop, checkpointing, and benchmark utilities |
| 6–8 | `search/objectives.py`, `search/encoding.py` | Multi-objective fitness, search-space encoding, and genetic operators |
| 9–11 | `search/algorithms/mopso.py`, `ga.py`, `pso.py` | MOPSO, GA, and PSO implementations |
| 12–13 | `search/pareto.py`, `search/algorithms/nsga2.py` | Pareto ranking and NSGA-II |
| 14–21 | `experiments/run_search.py`, `experiments/seed_runner.py`, `evaluation/benchmark.py` | Primary runs, final benchmarks, repeated-seed evaluation, and significance test |
| 22–27 | `data/cew.py`, `models/transfer_backbones.py`, `evaluation/metrics.py`, `experiments/run_transfer_baselines.py` | Transfer-backbone loading, CEW fine-tuning, and evaluation |

The notebooks in this release are gated launchers for these modules; the original merged notebook is not duplicated as a second executable implementation.

## Project layout

```text
.
├── README.md
├── CITATION.cff
├── pyproject.toml
├── requirements.txt
├── configs/
│   └── paper_default.yaml
├── notebooks/
│   ├── 01_optimizer_search.ipynb
│   └── 02_transfer_baselines.ipynb
├── src/mopso_drowsiness/
│   ├── config.py
│   ├── seeding.py
│   ├── data/
│   ├── models/
│   ├── search/
│   │   └── algorithms/
│   ├── training/
│   ├── evaluation/
│   └── experiments/
├── scripts/
├── tests/
└── .github/workflows/tests.yml
```

## Environment setup

Choose a PyTorch/torchvision build compatible with the target CPU/GPU environment, then install the listed dependencies and the local package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The default data locations match the original hosted-notebook paths. Set these variables when using another layout:

```bash
export MRL_TRAIN_ROOT=/path/to/mrl/train
export MRL_VAL_ROOT=/path/to/mrl/val
export MRL_TEST_ROOT=/path/to/mrl/test
export CEW_ROOT=/path/to/cew
export TRANSFER_WEIGHTS_DIR=/path/to/transfer/checkpoints
export DDD_OUTPUT_DIR=/path/to/output
```

Before a run, check the configured directories without starting training:

```bash
python scripts/preflight.py
```

The preflight reports missing roots and expected class folders. It does not fetch data or inspect checkpoint contents.

## Running experiments

The notebooks are thin entrypoints and have `RUN_EXPERIMENT = False` by default. Change that value only when the configured data, checkpoint paths, and compute budget are ready.

For the full optimizer workflow from the command line:

```bash
python scripts/run_search.py
```

To omit either final primary-model benchmarking or the repeated-seed test, select the corresponding option explicitly:

```bash
python scripts/run_search.py --skip-primary-benchmarks
python scripts/run_search.py --skip-seed-evaluation
```

For transfer-learning baselines:

```bash
python scripts/run_transfer_baselines.py
```

Outputs are written under `DDD_OUTPUT_DIR` (default: `outputs/`), including checkpoints and result tables. The command-line scripts launch actual training/evaluation; they have not been run as part of this code organization.

## Data, checkpoints, and reproducibility

Datasets and model checkpoints are intentionally excluded from this repository. Confirm the dataset versions, licenses, class-folder structure, split provenance, checkpoint identities, and redistribution permissions before running or publishing results. The code uses the split and label conventions present in the source notebook; this repository does not independently validate them.

The tests verify the configured seed schedules, notebook JSON/syntax and default execution gates, source-file comment policy, and project layout. The GitHub workflow compiles Python sources and runs those lightweight tests. Passing them does not validate data loading, model training, checkpoint compatibility, or the manuscript's reported results.

## Citation

Use the author and manuscript details in `CITATION.cff`. No DOI or final publication venue is recorded in this repository.

## License

No license file is included in this working project yet. A license must be selected by the authors before public release; without one, reuse permissions are not specified.
