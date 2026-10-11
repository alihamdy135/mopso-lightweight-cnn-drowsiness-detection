from ..config import *
from ..data.cew import get_cew_finetune_test_split
from ..data.mrl import EyeRegionDataset
from ..models.dynamic_cnn import SingleEyeRegionCNN

def _run_train_epoch(model: SingleEyeRegionCNN, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    """Returns (mean_loss, train_accuracy)."""
    model.train()
    _set_frozen_blocks_eval(model)

    total_loss = 0.0
    correct    = 0
    total      = 0

    for crop, labels in loader:
        crop   = crop.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out  = model(crop)
        loss = criterion(out, labels)
        loss.backward()

        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total_loss += loss.item() * labels.size(0)
        total      += labels.size(0)

    train_loss = total_loss / total if total else 0.0
    train_acc  = (correct / total * 100) if total else 0.0
    return train_loss, train_acc

def _compute_metrics(y_true, y_pred, y_probs) -> Dict:
    """Computes full suite of classification metrics, including RMSE between the
    predicted positive-class probability and the true (0/1) label."""
    if len(set(y_true)) < 2:
        auc_val = 0.0
    else:
        auc_val = roc_auc_score(y_true, y_probs) * 100

    y_true_arr  = np.asarray(y_true, dtype=float)
    y_probs_arr = np.asarray(y_probs, dtype=float)
    rmse_val = float(np.sqrt(np.mean((y_probs_arr - y_true_arr) ** 2))) if len(y_true_arr) else 0.0

    return {
        "balanced_acc": balanced_accuracy_score(y_true, y_pred) * 100,
        "F1": f1_score(y_true, y_pred, zero_division=0) * 100,
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "precision": precision_score(y_true, y_pred, zero_division=0) * 100,
        "recall": recall_score(y_true, y_pred, zero_division=0) * 100,
        "AUC": auc_val,
        "RMSE": rmse_val,
    }

def _evaluate(model: SingleEyeRegionCNN, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict:
    """Full evaluation pass including probability extraction for AUC."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0
    total      = 0

    for crop, labels in loader:
        crop   = crop.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        out   = model(crop)
        loss  = criterion(out, labels)

        probs = F.softmax(out, dim=1)[:, 1]
        preds = out.argmax(1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_loss += loss.item() * labels.size(0)
        total      += labels.size(0)

    m = _compute_metrics(all_labels, all_preds, all_probs)
    m["loss"] = total_loss / total if total else 0.0
    return m

def _set_frozen_blocks_eval(model: nn.Module) -> None:
    """Puts any block/layer with no trainable parameters into eval() mode (e.g. a frozen
    conv block), so its BatchNorm/Dropout behave correctly during partial fine-tuning.
    Generic across architectures: only iterates model.backbone if that attribute exists
    (SingleEyeRegionCNN, StandardVGG3, SimpleCNN). Architectures without a `.backbone`
    attribute (ResNet18, MobileNetV2, or any other torchvision model) are left untouched,
    since this notebook always trains them fully (no partial-freezing scheme is applied)."""
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return
    for blk in backbone:
        if not any(p.requires_grad for p in blk.parameters()):
            blk.eval()

def measure_inference_speed(model: nn.Module, device: torch.device, n_runs: int = 200, batch_size: int = 1):
    """Average per-image inference latency (ms) and FPS on a single representative batch."""
    model.eval()
    dummy = torch.randn(batch_size, 1, EYE_SIZE, EYE_SIZE, device=device)
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_runs):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
    ms_per_image = (t1 - t0) / n_runs / batch_size * 1000.0
    fps = 1000.0 / ms_per_image
    return ms_per_image, fps

def count_flops(model: nn.Module, input_size: Tuple[int, int, int, int] = None) -> int:
    """Estimates total FLOPs (2x MACs) for Conv2d and Linear layers via forward hooks.
    Pooling / activation / BatchNorm layers are cheap relative to conv / linear and are
    ignored, which is the standard convention for FLOPs reporting in NAS papers."""
    if input_size is None:
        input_size = (1, 1, EYE_SIZE, EYE_SIZE)

    flop_counts = []
    hooks = []

    def _conv_hook(module, inp, out):
        out_h, out_w = out.shape[2], out.shape[3]
        kh, kw = module.kernel_size
        in_c, out_c, groups = module.in_channels, module.out_channels, module.groups
        macs = (in_c // groups) * out_c * kh * kw * out_h * out_w
        flop_counts.append(2 * macs)

    def _linear_hook(module, inp, out):
        macs = module.in_features * module.out_features
        flop_counts.append(2 * macs)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_linear_hook))

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        model(torch.randn(*input_size, device=device))

    for h in hooks:
        h.remove()
    model.train(was_training)

    return int(sum(flop_counts))

def measure_peak_memory_mb(model: nn.Module, device: torch.device, batch_size: int = 1) -> float:
    """Measures peak GPU memory (MB) allocated during one forward pass (params + activations).
    Returns 0.0 on CPU, since torch.cuda memory stats aren't meaningful there."""
    if device.type != "cuda":
        return 0.0
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    dummy = torch.randn(batch_size, 1, EYE_SIZE, EYE_SIZE, device=device)
    with torch.no_grad():
        model(dummy)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)

def train_on_primary(model: nn.Module, train_loader, val_loader, device, epochs=PRIMARY_EPOCHS, lr=1e-3,
                      tag="model", resume=True):
    """Trains `model` on the PRIMARY dataset (MRL) for `epochs`. RESUMABLE: checkpoints
    every epoch to CKPT_DIR keyed by `tag`; a completed run for this exact tag returns
    instantly on the next call. (Was `train_on_cew` in the original notebook -- logic
    unchanged, only the dataset passed in by the caller has changed.)"""
    done_file = CKPT_DIR / f"train_done_{tag}.pth"
    state_file = CKPT_DIR / f"train_state_{tag}.pt"

    if resume and done_file.exists():
        model.load_state_dict(torch.load(done_file, map_location=device))
        log.info(f"  [{tag}] Found completed checkpoint -- skipping training ({epochs} epochs).")
        return model

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    start_epoch, best_acc, best_state = 1, 0.0, None

    if resume and state_file.exists():
        ck = torch.load(state_file, map_location=device)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optim_state"])
        start_epoch = ck["epoch"] + 1
        best_acc = ck["best_acc"]
        best_state = ck["best_state"]
        log.info(f"  [{tag}] Resuming primary (MRL) training at epoch {start_epoch}/{epochs} "
                 f"(best_acc so far={best_acc:.2f}%)")

    for epoch in range(start_epoch, epochs + 1):
        _run_train_epoch(model, train_loader, criterion, optimizer, device)
        val_m = _evaluate(model, val_loader, criterion, device)
        if val_m["accuracy"] > best_acc:
            best_acc = val_m["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
        if epoch % 10 == 0 or epoch == epochs:
            log.info(f"  [{tag}] MRL epoch {epoch}/{epochs} | val_acc={val_m['accuracy']:.2f}%")

        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "best_acc": best_acc, "best_state": best_state}, state_file)

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), done_file)
    return model

def finetune_on_secondary(model: nn.Module, ft_loader, device, epochs=FT_EPOCHS, lr=1e-4,
                           tag="model", resume=True):
    """Fine-tunes `model` on the SECONDARY dataset (CEW, deduplicated). RESUMABLE:
    checkpoints every epoch to CKPT_DIR keyed by `tag`. (Was `finetune_on_mrl` in the
    original notebook -- logic unchanged, only the dataset passed in has changed.)"""
    done_file = CKPT_DIR / f"finetune_done_{tag}.pth"
    state_file = CKPT_DIR / f"finetune_state_{tag}.pt"

    if resume and done_file.exists():
        model.load_state_dict(torch.load(done_file, map_location=device))
        log.info(f"  [{tag}] Found completed CEW fine-tune checkpoint -- skipping.")
        return model

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    start_epoch = 1

    if resume and state_file.exists():
        ck = torch.load(state_file, map_location=device)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optim_state"])
        start_epoch = ck["epoch"] + 1
        log.info(f"  [{tag}] Resuming CEW fine-tune at epoch {start_epoch}/{epochs}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        for imgs, labels in ft_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
        log.info(f"  [{tag}] CEW fine-tune epoch {epoch}/{epochs} complete")
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict()}, state_file)

    torch.save(model.state_dict(), done_file)
    return model

def full_benchmark(model: nn.Module, val_loader, device, tag="model"):
    """Computes the full metric suite (Acc/Prec/Rec/F1/AUC/Params/FLOPs/Inference/Memory)
    used for the final reported results."""
    criterion = nn.CrossEntropyLoss()
    m = _evaluate(model, val_loader, criterion, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ms_per_img, fps = measure_inference_speed(model, device)
    flops = count_flops(model)
    peak_memory_mb = measure_peak_memory_mb(model, device)
    return {
        "Model": tag,
        "Accuracy": m["accuracy"], "Precision": m["precision"], "Recall": m["recall"],
        "F1": m["F1"], "AUC": m["AUC"], "RMSE": m["RMSE"],
        "Parameters": n_params, "FLOPs": flops,
        "Inference_ms": ms_per_img, "FPS": fps,
        "Peak_Memory_MB": peak_memory_mb,
    }
