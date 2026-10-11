import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mopso_drowsiness.experiments.run_transfer_baselines import run_transfer_baselines


if __name__ == "__main__":
    run_transfer_baselines()
