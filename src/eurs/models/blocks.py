"""Only TensorRT friendly convolution, pooling, and resize blocks."""
import torch
from torch import nn
from torch.nn import functional as F


def channels(value, width):
    return max(8, int(round(value * width / 8)) * 8)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, groups=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride, kernel // 2, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )


class InvertedResidualDW(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expansion=3):
        super().__init__()
        mid = in_ch * expansion
        self.net = nn.Sequential(
            ConvBNReLU(in_ch, mid, 1),
            ConvBNReLU(mid, mid, 3, stride, groups=mid),
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.use_residual = stride == 1 and in_ch == out_ch
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.net(x)
        return self.act(x + y if self.use_residual else y)


class PPM(nn.Module):
    def __init__(self, in_ch, branch_ch=None):
        super().__init__()
        branch_ch = branch_ch or max(8, in_ch // 4)
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(s) for s in (1, 2, 4)])
        self.projections = nn.ModuleList([nn.Sequential(nn.Conv2d(in_ch, branch_ch, 1), nn.ReLU(inplace=True)) for _ in range(3)])
        self.fuse = ConvBNReLU(in_ch + 3 * branch_ch, in_ch, 1)

    def forward(self, x):
        size = x.shape[-2:]
        features = [x]
        for pool, projection in zip(self.pools, self.projections):
            features.append(F.interpolate(projection(pool(x)), size=size, mode="bilinear", align_corners=False))
        return self.fuse(torch.cat(features, dim=1))
