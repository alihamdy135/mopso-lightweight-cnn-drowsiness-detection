from ..config import *
from ..data.cew import EyeRegionDatasetRGB

def _adapt_first_conv_grayscale(model, first_conv_path):
    """Replaces a model's first conv layer with a 1-input-channel version, matching
    the training-time adaptation (ImageNet weights were 3-channel; grayscale eye crops
    are 1-channel)."""
    parts = first_conv_path.split('.')
    parent = model
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    attr_name = parts[-1]
    old_conv = parent[int(attr_name)] if attr_name.isdigit() else getattr(parent, attr_name)
    new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                          stride=old_conv.stride, padding=old_conv.padding,
                          bias=(old_conv.bias is not None))
    if attr_name.isdigit():
        parent[int(attr_name)] = new_conv
    else:
        setattr(parent, attr_name, new_conv)

class PretrainedBaseline(nn.Module):
    """Wrapper for ImageNet-pretrained backbones with a custom grayscale-adapted
    first conv and a custom 1- or 2-dense classifier head, attached as a SEPARATE
    top-level attribute (not nested inside the torchvision model's own head)."""
    def __init__(self, arch_name, n_dense_layers=1, num_classes=NUM_CLASSES):
        super().__init__()
        self.arch_name = arch_name
        self.n_dense_layers = n_dense_layers

        if arch_name == "vgg16":
            self.backbone = models.vgg16(weights=None)
            _adapt_first_conv_grayscale(self.backbone, 'features.0')


            self.backbone.avgpool = nn.AdaptiveAvgPool2d(1)
            feat_dim = 512
            self.backbone.classifier = nn.Identity()
            self.pool = nn.Identity()
            self.flatten_dim = feat_dim

        elif arch_name == "resnet50":
            self.backbone = models.resnet50(weights=None)
            _adapt_first_conv_grayscale(self.backbone, 'conv1')
            feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.pool = nn.Identity()
            self.flatten_dim = feat_dim

        elif arch_name == "mobilenet_v2":
            self.backbone = models.mobilenet_v2(weights=None)
            _adapt_first_conv_grayscale(self.backbone, 'features.0.0')
            feat_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.flatten_dim = feat_dim

        else:
            raise ValueError(f"Unknown architecture: {arch_name}")

        if n_dense_layers == 1:
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Dropout(0.5), nn.Linear(self.flatten_dim, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Dropout(0.5), nn.Linear(self.flatten_dim, 256),
                nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes)
            )

    def forward(self, x):
        feats = self.backbone(x)
        if feats.dim() == 4:
            feats = self.pool(feats)
        return self.classifier(feats)

def build_backbone(model_name: str) -> nn.Module:
    if model_name == "MobileNetV2_1dense": return PretrainedBaseline("mobilenet_v2", 1)
    if model_name == "MobileNetV2_2dense": return PretrainedBaseline("mobilenet_v2", 2)
    if model_name == "ResNet50_1dense":    return PretrainedBaseline("resnet50", 1)
    if model_name == "ResNet50_2dense":    return PretrainedBaseline("resnet50", 2)
    if model_name == "VGG16_1dense":       return PretrainedBaseline("vgg16", 1)
    if model_name == "VGG16_2dense":       return PretrainedBaseline("vgg16", 2)

    raise ValueError(f"Unknown model_name: {model_name}")

def _finetune_lr_for(model, base_lr):
    """VGG16 has no batch normalization (plain conv+ReLU stack), unlike ResNet50/
    MobileNetV2, which makes it far more prone to collapsing to a constant prediction
    at a learning rate that's stable for the BN-equipped architectures. Use a lower LR
    for it specifically, matching the fix applied in the training notebooks."""
    arch_name = getattr(model, "arch_name", "")
    return base_lr / 10 if arch_name == "vgg16" else base_lr

def finetune_on_cew(model: nn.Module, train_loader: DataLoader, device: torch.device,
                     epochs: int = CEW_FINETUNE_EPOCHS, lr: float = CEW_FINETUNE_LR) -> nn.Module:
    """Fine-tunes `model` (already loaded with MRL-pretrained weights) on the CEW
    fine-tune split for a fixed number of epochs -- no early stopping, no validation
    split, matching a straightforward "N epochs then test" protocol. Gradient clipping
    is applied throughout, which matters most for VGG16's lack of batch norm."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=_finetune_lr_for(model, lr), weight_decay=1e-3)

    for epoch in range(1, epochs + 1):
        total_loss, n_seen = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            n_seen += imgs.size(0)
        log.info(f"    [CEW finetune] epoch {epoch}/{epochs} -- loss={total_loss / n_seen:.4f}")

    return model
