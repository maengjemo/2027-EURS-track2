"""EURS v0 joint restoration and segmentation model."""
import torch
from torch import nn
from torch.nn import functional as F

from .blocks import ConvBNReLU, InvertedResidualDW, PPM, channels


class DecoderRefine(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(ConvBNReLU(ch, ch, 3, groups=ch), ConvBNReLU(ch, ch, 1))

    def forward(self, x):
        return self.net(x)


class EURSv0(nn.Module):
    """Shared encoder, independent FPN segmentation and RGB correction heads.

    Width 1.0: 592,604 trainable PyTorch parameters. At 512×1024, the
    organizer's ONNX checker measures 0.5802M initializers and 2.7714 GFLOPs.
    """

    def __init__(self, width=1.0):
        super().__init__()
        c1, c2, c3, c4, c5 = [channels(v, width) for v in (16, 24, 48, 96, 128)]
        dec, local = channels(32, width), channels(16, width)
        self.stem = ConvBNReLU(3, c1, 3, 2)
        self.stage1 = self._stage(c1, c2, 2)
        self.stage2 = self._stage(c2, c3, 3)
        self.stage3 = self._stage(c3, c4, 4)
        self.stage4 = self._stage(c4, c5, 3)
        self.ppm = PPM(c5)
        self.medium = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c5, 64), nn.ReLU(inplace=True))
        self.water_head = nn.Linear(64, 10)
        self.depth_head = nn.Linear(64, 1)
        self.film = nn.Linear(64, dec * 2)
        self.laterals = nn.ModuleList([nn.Conv2d(c, dec, 1) for c in (c2, c3, c4, c5)])
        self.refines = nn.ModuleList([DecoderRefine(dec) for _ in range(4)])
        self.seg_head = nn.Conv2d(dec, 5, 1)
        self.global_rgb = nn.Linear(64, 9)
        self.local = nn.Sequential(ConvBNReLU(c2, local, 1), ConvBNReLU(local, local, 3, groups=local), nn.Conv2d(local, 3, 1))

    @staticmethod
    def _stage(in_ch, out_ch, count):
        return nn.Sequential(InvertedResidualDW(in_ch, out_ch, 2), *[InvertedResidualDW(out_ch, out_ch) for _ in range(count - 1)])

    def forward(self, x):
        h, w = x.shape[-2:]
        y = self.stem(x)
        f1 = self.stage1(y)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.ppm(self.stage4(f3))
        medium = self.medium(f4)
        film = self.film(medium)
        gamma, beta = film.chunk(2, dim=1)
        gamma = torch.sigmoid(gamma).unsqueeze(-1).unsqueeze(-1) * 2
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        features = (f1, f2, f3, f4)
        seg = None
        for i in (3, 2, 1, 0):
            skip = self.laterals[i](features[i])
            seg = skip if seg is None else skip + F.interpolate(seg, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            seg = self.refines[i](seg * gamma + beta)
        seg_logits = F.interpolate(self.seg_head(seg), size=(h, w), mode="bilinear", align_corners=False)
        rgb = self.global_rgb(medium).reshape(-1, 9, 1, 1)
        gain = 0.5 + torch.sigmoid(rgb[:, 0:3])
        bias = (torch.sigmoid(rgb[:, 3:6]) - 0.5) * 0.2
        gamma_mix = torch.sigmoid(rgb[:, 6:9])
        corrected = x * gain + bias
        corrected = corrected * (1 - gamma_mix) + corrected * corrected * gamma_mix
        residual = F.interpolate(self.local(f1), size=(h, w), mode="bilinear", align_corners=False)
        restored = torch.clamp(corrected + residual, 0.0, 1.0)
        return {"restored_image": restored, "seg_logits": seg_logits,
                "water_logits": self.water_head(medium), "depth": self.depth_head(medium)}

    def forward_export(self, x):
        outputs = self.forward(x)
        return outputs["restored_image"], outputs["seg_logits"]


class ExportModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward_export(x)
