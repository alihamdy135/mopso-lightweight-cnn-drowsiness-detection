import json
import os
from pathlib import Path

DEFAULTS = {
    "MRL_TRAIN_ROOT": "/kaggle/input/datasets/akashshingha850/mrl-eye-dataset/data/train",
    "MRL_VAL_ROOT": "/kaggle/input/datasets/akashshingha850/mrl-eye-dataset/data/val",
    "MRL_TEST_ROOT": "/kaggle/input/datasets/akashshingha850/mrl-eye-dataset/data/test",
    "CEW_ROOT": "/kaggle/input/datasets/rashammmm/datasetcew/dataset_B_Eye_Images",
    "TRANSFER_WEIGHTS_DIR": "/kaggle/input/datasets/laurentamer/eye-state-model-weights",
}

EXPECTED_CHILDREN = {
    "MRL_TRAIN_ROOT": ["sleepy", "awake"],
    "MRL_VAL_ROOT": ["sleepy", "awake"],
    "MRL_TEST_ROOT": ["sleepy", "awake"],
    "CEW_ROOT": ["closedLeftEyes", "closedRightEyes", "openLeftEyes", "openRightEyes"],
}

checks = {}
for name, default in DEFAULTS.items():
    root = Path(os.environ.get(name, default)).expanduser()
    required_children = EXPECTED_CHILDREN.get(name, [])
    checks[name] = {
        "path": str(root),
        "exists": root.is_dir(),
        "missing_children": [child for child in required_children if not (root / child).is_dir()],
    }
print(json.dumps(checks, indent=2))
raise SystemExit(0 if all(value["exists"] and not value["missing_children"] for value in checks.values()) else 1)
