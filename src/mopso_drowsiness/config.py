import copy
import hashlib
import json
import logging
import os
import random
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from scipy.stats import spearmanr, ttest_rel, wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_CONFIG_PATH = PROJECT_ROOT / "configs" / "paper_default.yaml"
PAPER_CONFIG = yaml.safe_load(PAPER_CONFIG_PATH.read_text(encoding="utf-8"))
SEARCH_SPACE = PAPER_CONFIG["search_space"]
SEARCH_CONFIG = PAPER_CONFIG["search"]
TRAINING_CONFIG = PAPER_CONFIG["training"]
TRANSFER_CONFIG = PAPER_CONFIG["transfer"]

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("mopso_drowsiness")

SS_FILTERS = SEARCH_SPACE["filters"]
SS_KERNELS = SEARCH_SPACE["kernels"]
SS_ACTIVATIONS = SEARCH_SPACE["activations"]
SS_DROPOUT_CONV = SEARCH_SPACE["dropout_conv"]
SS_BN_PLACEMENT = SEARCH_SPACE["batch_norm_placement"]
SS_DEPTHS = SEARCH_SPACE["depths"]
SS_DENSE_UNITS = SEARCH_SPACE["dense_units"]
SS_DROPOUT_DENSE = SEARCH_SPACE["dropout_dense"]

GA_POP_SIZE = SEARCH_CONFIG["population_size"]
GA_GENERATIONS = SEARCH_CONFIG["generations"]
GA_EPOCHS_PER_EVAL = SEARCH_CONFIG["epochs_per_evaluation"]
GA_PATIENCE = SEARCH_CONFIG["patience"]
SEARCH_PROXY_FRAC = SEARCH_CONFIG["proxy_fraction"]
GA_BATCH = SEARCH_CONFIG["batch_size"]
GA_LR = SEARCH_CONFIG["learning_rate"]
GA_CROSSOVER_RATE = SEARCH_CONFIG["crossover_rate"]
GA_MUT_PROB = SEARCH_CONFIG["mutation_probability"]
GA_TOURNAMENT_K = SEARCH_CONFIG["tournament_size"]
GA_ELITISM = SEARCH_CONFIG["elitism"]
PSO_W = SEARCH_CONFIG["pso_inertia"]
PSO_C1 = SEARCH_CONFIG["pso_cognitive"]
PSO_C2 = SEARCH_CONFIG["pso_social"]
GA_SWARM_SIZE = GA_POP_SIZE
GA_ARCHIVE_SIZE = SEARCH_CONFIG["archive_size"]
GA_N_DIVISIONS = SEARCH_CONFIG["divisions"]
N_STAT_RUNS = SEARCH_CONFIG["statistical_runs"]
SEARCH_SEED = SEARCH_CONFIG["primary_seed"]
NSGA2_SEARCH_SEED = SEARCH_CONFIG["nsga2_primary_seed"]
GA_SEED_START = SEARCH_CONFIG["repeated_seed_start"]
NSGA2_SEED_START = SEARCH_CONFIG["nsga2_seed_start"]

PRIMARY_EPOCHS = TRAINING_CONFIG["primary_epochs"]
FT_EPOCHS = TRAINING_CONFIG["secondary_epochs"]
CEW_EPOCHS = PRIMARY_EPOCHS
MRL_FT_EPOCHS = FT_EPOCHS
EYE_SIZE = TRAINING_CONFIG["eye_size"]
CEW_FINETUNE_EPOCHS = FT_EPOCHS
CEW_FINETUNE_LR = TRAINING_CONFIG["cew_finetune_learning_rate"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MRL_TRAIN_ROOT = os.environ.get("MRL_TRAIN_ROOT", "/kaggle/input/datasets/akashshingha850/mrl-eye-dataset/data/train")
MRL_VAL_ROOT = os.environ.get("MRL_VAL_ROOT", "/kaggle/input/datasets/akashshingha850/mrl-eye-dataset/data/val")
MRL_TEST_ROOT = os.environ.get("MRL_TEST_ROOT", "/kaggle/input/datasets/akashshingha850/mrl-eye-dataset/data/test")
MRL_CLASS_MAP = {"sleepy": 0, "awake": 1}
CEW_ROOT = os.environ.get("CEW_ROOT", "/kaggle/input/datasets/rashammmm/datasetcew/dataset_B_Eye_Images")
CEW_SUBFOLDERS = {"closed": ["closedLeftEyes", "closedRightEyes"], "open": ["openLeftEyes", "openRightEyes"]}
CEW_FINETUNE_FRAC = SEARCH_CONFIG.get("cew_finetune_fraction", TRANSFER_CONFIG["cew_finetune_fraction"])
CEW_SPLIT_SEED = TRANSFER_CONFIG["cew_split_seed"]
NUM_CLASSES = TRANSFER_CONFIG["num_classes"]
BATCH_SIZE = TRANSFER_CONFIG["batch_size"]
MODEL_NAMES = TRANSFER_CONFIG["model_names"]
MODEL_WEIGHT_FOLDERS = TRANSFER_CONFIG["model_weight_folders"]

OUTPUT_ROOT = Path(os.environ.get("DDD_OUTPUT_DIR", str(PROJECT_ROOT / "outputs"))).expanduser()
CKPT_DIR = Path(os.environ.get("DDD_CHECKPOINT_DIR", str(OUTPUT_ROOT / "checkpoints"))).expanduser()
RESULTS_DIR = Path(os.environ.get("DDD_RESULTS_DIR", str(OUTPUT_ROOT / "results"))).expanduser()
EXTRACT_DIR = Path(os.environ.get("DDD_EXTRACT_DIR", str(OUTPUT_ROOT / "weights_extracted"))).expanduser()
REZIP_CACHE_DIR = Path(os.environ.get("DDD_REZIP_CACHE_DIR", str(OUTPUT_ROOT / "rezipped_checkpoints"))).expanduser()
GA_CKPT_PATH = Path(os.environ.get("DDD_GENOME_PATH", str(CKPT_DIR / "best_mopso_genome.json"))).expanduser()
WEIGHTS_DIR = Path(os.environ.get("TRANSFER_WEIGHTS_DIR", "/kaggle/input/datasets/laurentamer/eye-state-model-weights")).expanduser()

__all__ = [name for name in globals() if not name.startswith("_")]
