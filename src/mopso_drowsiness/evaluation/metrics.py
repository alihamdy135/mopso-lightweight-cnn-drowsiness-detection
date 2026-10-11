from ..config import *

def measure_inference_speed(model: nn.Module, device: torch.device, n_runs: int = 200, batch_size: int = 1):
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
