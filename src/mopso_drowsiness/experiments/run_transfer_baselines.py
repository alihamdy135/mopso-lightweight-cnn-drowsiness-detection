from ..config import *
from ..data.cew import EyeRegionDatasetRGB, get_cew_train_test_split
from ..evaluation.metrics import count_flops, measure_inference_speed, measure_peak_memory_mb
from ..models.transfer_backbones import PretrainedBaseline, build_backbone, finetune_on_cew

_WEIGHT_EXTS = (".pt", ".pth", ".bin", ".ckpt")


def _rezip_exploded_checkpoint(payload_root: Path, out_path: Path) -> Path:
    """Re-packs an exploded PyTorch zip-checkpoint folder (one that contains data.pkl,
    version, byteorder, data/, .data/, .format_version, .storage_alignment directly
    inside it) back into a loadable .pt zip archive at `out_path`, preserving the
    original internal root-folder name and every file exactly (ZIP_STORED, no
    compression, matching PyTorch's own on-disk format)."""
    root_name = payload_root.name
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_STORED) as zf:
        for f in payload_root.rglob("*"):
            if f.is_file():
                arcname = f"{root_name}/{f.relative_to(payload_root)}"
                zf.write(f, arcname)
    return out_path

def resolve_checkpoint(folder: Path) -> Path:
    """Given a folder, returns a path torch.load() can open directly. Handles:
      1) a loose .zip sitting directly inside `folder` (extracted in place),
      2) an already-intact weight file (.pt/.pth/.bin/.ckpt) anywhere inside,
      3) an exploded PyTorch zip-checkpoint (data.pkl found somewhere inside, with no
         intact archive file) -- re-zipped into REZIP_CACHE_DIR and returned.
    Raises FileNotFoundError if none of the above is found."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Weight folder does not exist: {folder}")

    for zpath in folder.glob("*.zip"):
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(folder)
        log.info(f"  Extracted {zpath.name} inside {folder.name}")

    candidates = [f for f in folder.rglob("*") if f.suffix.lower() in _WEIGHT_EXTS]
    if candidates:
        if len(candidates) > 1:
            log.warning(f"[Weights] Multiple weight files inside {folder}: "
                        f"{[c.name for c in candidates]} -- using the first one.")
        return candidates[0]

    pkl_matches = list(folder.rglob("data.pkl"))
    if not pkl_matches:
        raise FileNotFoundError(
            f"No weight file (.pt/.pth/.bin/.ckpt) and no data.pkl (exploded checkpoint) "
            f"found anywhere inside {folder}. Check the folder actually contains the "
            f"checkpoint (run: `!find {folder} -type f` in a cell to list its contents).")
    if len(pkl_matches) > 1:
        log.warning(f"[Weights] Multiple data.pkl found inside {folder}: "
                    f"{[str(p) for p in pkl_matches]} -- using the first one.")

    payload_root = pkl_matches[0].parent
    out_path = REZIP_CACHE_DIR / f"{folder.name}.pt"
    if not out_path.exists():
        _rezip_exploded_checkpoint(payload_root, out_path)
        log.info(f"  [Weights] Re-zipped exploded checkpoint {payload_root} -> {out_path.name}")
    return out_path

def resolve_all_folders(weights_dir: Path, model_folders: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Path]]:
    """Turns MODEL_WEIGHT_FOLDERS entries into resolved, loadable checkpoint paths.
    Returns manifest[model_name][stage] = checkpoint_path (only for stages that were
    given and resolved successfully)."""
    weights_dir = Path(weights_dir)
    manifest: Dict[str, Dict[str, Path]] = {}

    for model_name, stages in model_folders.items():
        manifest[model_name] = {}
        for stage, folder_value in stages.items():
            if folder_value is None:
                log.info(f"[Weights] {model_name} [{stage}]: not provided -- skipping.")
                continue
            folder_path = Path(folder_value)
            if not folder_path.is_absolute():
                folder_path = weights_dir / folder_path
            try:
                ckpt_path = resolve_checkpoint(folder_path)
                manifest[model_name][stage] = ckpt_path
                log.info(f"[Weights] {model_name} [{stage}] -> {ckpt_path}")
            except FileNotFoundError as e:
                log.warning(f"[Weights] {model_name} [{stage}]: {e}")

    return manifest

def load_weights_into(model: nn.Module, weight_path: Path, device: torch.device) -> nn.Module:
    """Loads a checkpoint that may be a raw state_dict, or a dict containing one under a
    common key ('state_dict', 'model_state', 'model_state_dict'), or a fully pickled model
    object. Reports missing/unexpected keys so a head/trunk mismatch is easy to spot."""
    ckpt = torch.load(weight_path, map_location=device)

    if isinstance(ckpt, nn.Module):
        return ckpt.to(device)

    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        log.warning(f"  [{weight_path.name}] Missing keys ({len(missing)}): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        log.warning(f"  [{weight_path.name}] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")
    if not missing and not unexpected:
        log.info(f"  [{weight_path.name}] State dict loaded cleanly, no key mismatches.")

    return model.to(device)

def evaluate_on(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()


    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    all_preds, all_labels, all_probs = [], [], []
    total_images = 0


    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()


    for imgs, labels in loader:
        imgs = imgs.to(device)
        out = model(imgs)
        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(1).cpu().numpy()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        total_images += imgs.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()


    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0


    total_ms = (t1 - t0) * 1000.0
    ms_per_img = total_ms / total_images if total_images > 0 else 0.0
    fps = 1000.0 / ms_per_img if ms_per_img > 0 else 0.0

    auc_val = roc_auc_score(all_labels, all_probs) * 100 if len(set(all_labels)) > 1 else 0.0

    return {
        "Accuracy": accuracy_score(all_labels, all_preds) * 100,
        "Precision": precision_score(all_labels, all_preds, zero_division=0) * 100,
        "Recall": recall_score(all_labels, all_preds, zero_division=0) * 100,
        "F1": f1_score(all_labels, all_preds, zero_division=0) * 100,
        "AUC": auc_val,
        "Inference_ms": ms_per_img,
        "FPS": fps,
        "Peak_Memory_MB": peak_mem_mb,
    }

def run_transfer_baselines() -> pd.DataFrame:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    REZIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    weight_manifest = resolve_all_folders(WEIGHTS_DIR, MODEL_WEIGHT_FOLDERS)
    mrl_test_ds = EyeRegionDatasetRGB(MRL_TEST_ROOT, {key: [key] for key in MRL_CLASS_MAP}, MRL_CLASS_MAP)
    cew_train_ds, cew_test_ds = get_cew_train_test_split()
    mrl_test_loader = DataLoader(mrl_test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    cew_train_loader = DataLoader(cew_train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    cew_test_loader = DataLoader(cew_test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    results = []

    for model_name in MODEL_NAMES:
        stage_paths = weight_manifest.get(model_name, {})
        mrl_wpath = stage_paths.get("mrl_pretrained")
        if mrl_wpath is None:
            log.warning(f"[Eval] No resolved mrl_pretrained checkpoint for '{model_name}' -- skipping.")
            continue


        probe_model = build_backbone(model_name)
        n_params = sum(p.numel() for p in probe_model.parameters() if p.requires_grad)
        flops = count_flops(probe_model)
        del probe_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


        log.info(f"\n=== {model_name} | stage=mrl_pretrained -> MRL test ===")
        model = build_backbone(model_name)
        model = load_weights_into(model, mrl_wpath, DEVICE)

        mrl_metrics = evaluate_on(model, mrl_test_loader, DEVICE)
        results.append({
            "Model": model_name, "Dataset": "MRL", "Stage": "mrl_pretrained",
            **mrl_metrics, "Parameters": n_params, "FLOPs": flops,
        })
        log.info(f"  Acc={mrl_metrics['Accuracy']:.2f}% | Latency={mrl_metrics['Inference_ms']:.3f}ms | PeakMem={mrl_metrics['Peak_Memory_MB']:.2f}MB")


        log.info(f"\n=== {model_name} | fine-tuning on CEW for {CEW_FINETUNE_EPOCHS} epochs -> CEW test ===")
        model = finetune_on_cew(model, cew_train_loader, DEVICE, epochs=CEW_FINETUNE_EPOCHS, lr=CEW_FINETUNE_LR)

        cew_metrics = evaluate_on(model, cew_test_loader, DEVICE)
        results.append({
            "Model": model_name, "Dataset": "CEW", "Stage": "cew_finetuned",
            **cew_metrics, "Parameters": n_params, "FLOPs": flops,
        })
        log.info(f"  Acc={cew_metrics['Accuracy']:.2f}% | Latency={cew_metrics['Inference_ms']:.3f}ms | PeakMem={cew_metrics['Peak_Memory_MB']:.2f}MB")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(results)

    results_df.to_csv(RESULTS_DIR / "transfer_backbone_benchmark.csv", index=False)

    print(results_df.to_string(index=False))
    return results_df
