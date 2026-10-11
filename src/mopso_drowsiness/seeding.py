def seed_schedule(base_seed: int, n_runs: int = 10) -> list[int]:
    if n_runs < 1:
        raise ValueError("n_runs must be positive")
    return [base_seed + run for run in range(n_runs)]


def set_all_seeds(seed: int = 42) -> None:
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f" All random seeds locked to {seed}")
