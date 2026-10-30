"""CGNet (Wu et al. 2020, arXiv:1811.08201) M3N21 순수 torch 자체 구현.

Context Guided(CG) block: local feature(depthwise 3x3) + surrounding context(depthwise
dilated 3x3)를 concat한 joint feature에 global context(SE)를 곱한다. 입력 이미지를
스테이지마다 다운샘플해 주입(InputInjection)한다. 구성은 M=3, N=21이다.

인터페이스: forward(x) -> dict(seg_logits=(B,5,H,W), restored_image=x). 복원은 Identity다.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ConvBNPReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel, stride=1, dilation=1):
        pad = (kernel // 2) * dilation
        super().__init__(nn.Conv2d(in_ch, out_ch, kernel, stride, pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_ch), nn.PReLU(out_ch))


class BNPReLU(nn.Sequential):
    def __init__(self, ch):
        super().__init__(nn.BatchNorm2d(ch), nn.PReLU(ch))


class FGlo(nn.Module):
    """Global context: GAP → FC bottleneck → sigmoid로 채널 가중(SE)."""

    def __init__(self, ch, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ch, max(1, ch // reduction)), nn.ReLU(inplace=True),
                                nn.Linear(max(1, ch // reduction), ch), nn.Sigmoid())

    def forward(self, x):
        b, c = x.shape[:2]
        y = x.mean(dim=(2, 3))
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ContextGuidedBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=2, reduction=16):
        super().__init__()
        mid = out_ch // 2
        self.reduce = ConvBNPReLU(in_ch, mid, 1)
        self.f_loc = nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False)
        self.f_sur = nn.Conv2d(mid, mid, 3, padding=dilation, dilation=dilation, groups=mid, bias=False)
        self.bn_act = BNPReLU(2 * mid)
        self.glo = FGlo(out_ch, reduction)
        self.use_residual = in_ch == out_ch

    def forward(self, x):
        y = self.reduce(x)
        joint = torch.cat([self.f_loc(y), self.f_sur(y)], dim=1)
        joint = self.bn_act(joint)
        joint = self.glo(joint)
        return x + joint if self.use_residual else joint


class ContextGuidedBlockDown(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=2, reduction=16):
        super().__init__()
        self.reduce = ConvBNPReLU(in_ch, out_ch, 3, stride=2)
        self.f_loc = nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False)
        self.f_sur = nn.Conv2d(out_ch, out_ch, 3, padding=dilation, dilation=dilation, groups=out_ch, bias=False)
        self.bn_act = BNPReLU(2 * out_ch)
        self.fuse = nn.Conv2d(2 * out_ch, out_ch, 1, bias=False)
        self.glo = FGlo(out_ch, reduction)

    def forward(self, x):
        y = self.reduce(x)
        joint = torch.cat([self.f_loc(y), self.f_sur(y)], dim=1)
        joint = self.fuse(self.bn_act(joint))
        return self.glo(joint)


class InputInjection(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pools = nn.ModuleList([nn.AvgPool2d(3, stride=2, padding=1) for _ in range(ratio)])

    def forward(self, x):
        for pool in self.pools:
            x = pool(x)
        return x


class CGNet(nn.Module):
    def __init__(self, num_classes=5, m=3, n=21, **kwargs):
        super().__init__()
        self.level1_0 = ConvBNPReLU(3, 32, 3, stride=2)
        self.level1_1 = ConvBNPReLU(32, 32, 3)
        self.level1_2 = ConvBNPReLU(32, 32, 3)
        self.sample1 = InputInjection(1)
        self.sample2 = InputInjection(2)
        self.b1 = BNPReLU(32 + 3)
        self.level2_0 = ContextGuidedBlockDown(32 + 3, 64, dilation=2, reduction=8)
        self.level2 = nn.ModuleList([ContextGuidedBlock(64, 64, dilation=2, reduction=8) for _ in range(m - 1)])
        self.bn_prelu_2 = BNPReLU(128 + 3)
        self.level3_0 = ContextGuidedBlockDown(128 + 3, 128, dilation=4, reduction=16)
        self.level3 = nn.ModuleList([ContextGuidedBlock(128, 128, dilation=4, reduction=16) for _ in range(n - 1)])
        self.bn_prelu_3 = BNPReLU(256)
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        inp1, inp2 = self.sample1(x), self.sample2(x)
        y = self.level1_2(self.level1_1(self.level1_0(x)))
        y = self.b1(torch.cat([y, inp1], dim=1))
        down2 = self.level2_0(y)
        y = down2
        for layer in self.level2:
            y = layer(y)
        y = self.bn_prelu_2(torch.cat([y, down2, inp2], dim=1))
        down3 = self.level3_0(y)
        y = down3
        for layer in self.level3:
            y = layer(y)
        y = self.bn_prelu_3(torch.cat([down3, y], dim=1))
        seg_logits = F.interpolate(self.classifier(y), size=(h, w), mode="bilinear", align_corners=False)
        return {"seg_logits": seg_logits, "restored_image": x}

    def forward_export(self, x):
        outputs = self.forward(x)
        return outputs["restored_image"], outputs["seg_logits"]
