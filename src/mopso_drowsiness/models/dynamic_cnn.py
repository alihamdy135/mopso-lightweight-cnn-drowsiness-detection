from ..config import *

class SingleEyeRegionCNN(nn.Module):
    def __init__(self, genome: Dict, num_classes=2):
        super().__init__()
        layers = []
        in_chs = 1
        curr_size = EYE_SIZE

        for block in genome['blocks']:
            if curr_size <= 2: break

            layers.append(nn.Conv2d(in_chs, block['filters'], kernel_size=block['kernel'], padding=block['kernel']//2))
            if block['bn_placement'] == 'before': layers.append(nn.BatchNorm2d(block['filters']))
            layers.append(getattr(nn, block['activation'])())
            if block['bn_placement'] == 'after': layers.append(nn.BatchNorm2d(block['filters']))

            layers.append(nn.MaxPool2d(2))
            layers.append(nn.Dropout2d(block['dropout']))

            in_chs = block['filters']
            curr_size //= 2

        self.backbone = nn.Sequential(*layers)
        self.flat_feats = in_chs * (curr_size * curr_size)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_feats, genome['dense_units']),
            getattr(nn, genome['dense_act'])(),
            nn.Dropout(genome['dense_drop']),
            nn.Linear(genome['dense_units'], num_classes)
        )

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        return self.classifier(self.backbone(x))
