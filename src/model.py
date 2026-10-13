"""SAIRANet: EfficientNet-B3 backbone + 4 concern heads + 1 skin-type head."""
import timm
import torch
import torch.nn as nn

from config import HEADS, SKIN_TYPES


def _head(in_dim, out_dim, p=0.3):
    return nn.Sequential(nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(p), nn.Linear(256, out_dim))


class SAIRANet(nn.Module):
    def __init__(self, pretrained=True, freeze_blocks=5, p=0.3):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b3", pretrained=pretrained, num_classes=0)
        d = self.backbone.num_features  # 1536
        self.drop = nn.Dropout(p)
        self.heads = nn.ModuleDict({h: _head(d, 1) for h in HEADS})
        self.skin_type = _head(d, len(SKIN_TYPES))
        self.freeze(freeze_blocks)

    def freeze(self, n_blocks):
        """Freeze the stem and the first n_blocks of the backbone."""
        for m in [self.backbone.conv_stem, self.backbone.bn1, *list(self.backbone.blocks)[:n_blocks]]:
            for prm in m.parameters():
                prm.requires_grad = False

    def forward(self, x):
        f = self.drop(self.backbone(x))
        out = {h: self.heads[h](f).squeeze(-1) for h in HEADS}  # logits
        out["skin_type"] = self.skin_type(f)
        return out

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        o = self(x)
        res = {h: torch.sigmoid(o[h]) for h in HEADS}
        res["skin_type"] = torch.softmax(o["skin_type"], dim=-1)
        return res
