import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mopso_drowsiness.experiments.run_search import run_search_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-primary-benchmarks", action="store_true")
    parser.add_argument("--skip-seed-evaluation", action="store_true")
    args = parser.parse_args()
    run_search_experiment(
        include_primary_benchmarks=not args.skip_primary_benchmarks,
        include_seed_evaluation=not args.skip_seed_evaluation,
    )


if __name__ == "__main__":
    main()
