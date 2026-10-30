"""ENet (Paszke et al. 2016, arXiv:1606.02147) 순수 torch 자체 구현.

논문 Table 1의 스테이지 구성을 따른다: initial block, 인코더 stage 1~3(bottleneck),
디코더 stage 4~5, full-conv. 원 논문은 다운샘플에서 MaxPool 인덱스를 저장하고 디코더에서
MaxUnpool로 복원하지만, ONNX/TensorRT에서 MaxUnpool은 export가 불안정하므로 이 구현은
디코더 업샘플을 bilinear + 1x1 conv 변형으로 대체한다(README에 표기). 그 외 bottleneck
구조(projection→conv→expansion, PReLU, spatial dropout)는 논문을 따른다.

인터페이스: forward(x) -> dict(seg_logits=(B,5,H,W), restored_image=x). 복원은 Identity다.
"""
import torch
from torch import nn
from torch.nn import functional as F


class InitialBlock(nn.Module):
    """3x3 stride-2 conv(3->out-3)와 MaxPool(3)을 concat해 out 채널을 만든다."""

    def __init__(self, out_ch=16):
        super().__init__()
        self.conv = nn.Conv2d(3, out_ch - 3, 3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU(out_ch)

    def forward(self, x):
        out = torch.cat([self.conv(x), self.pool(x)], dim=1)
        return self.act(self.bn(out))


class RegularBottleneck(nn.Module):
    def __init__(self, ch, internal_ratio=4, kernel=3, padding=1, dilation=1,
                 asymmetric=False, dropout=0.1):
        super().__init__()
        internal = ch // internal_ratio
        self.proj = nn.Sequential(nn.Conv2d(ch, internal, 1, bias=False),
                                  nn.BatchNorm2d(internal), nn.PReLU(internal))
        if asymmetric:
            conv = nn.Sequential(
                nn.Conv2d(internal, internal, (kernel, 1), padding=(padding, 0), bias=False),
                nn.BatchNorm2d(internal), nn.PReLU(internal),
                nn.Conv2d(internal, internal, (1, kernel), padding=(0, padding), bias=False),
                nn.BatchNorm2d(internal), nn.PReLU(internal))
        else:
            conv = nn.Sequential(
                nn.Conv2d(internal, internal, kernel, padding=padding, dilation=dilation, bias=False),
                nn.BatchNorm2d(internal), nn.PReLU(internal))
        self.conv = conv
        self.expand = nn.Sequential(nn.Conv2d(internal, ch, 1, bias=False), nn.BatchNorm2d(ch))
        self.drop = nn.Dropout2d(dropout)
        self.act = nn.PReLU(ch)

    def forward(self, x):
        y = self.drop(self.expand(self.conv(self.proj(x))))
        return self.act(x + y)


class DownsamplingBottleneck(nn.Module):
    """MaxPool 인덱스 없이 다운샘플한다(export 친화). main 분기는 pool + 1x1 채널 확장."""

    def __init__(self, in_ch, out_ch, internal_ratio=4, dropout=0.1):
        super().__init__()
        internal = out_ch // internal_ratio
        self.pool = nn.MaxPool2d(2, stride=2)
        self.main_conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.proj = nn.Sequential(nn.Conv2d(in_ch, internal, 2, stride=2, bias=False),
                                  nn.BatchNorm2d(internal), nn.PReLU(internal))
        self.conv = nn.Sequential(nn.Conv2d(internal, internal, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(internal), nn.PReLU(internal))
        self.expand = nn.Sequential(nn.Conv2d(internal, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.drop = nn.Dropout2d(dropout)
        self.act = nn.PReLU(out_ch)

    def forward(self, x):
        main = self.main_conv(self.pool(x))
        y = self.drop(self.expand(self.conv(self.proj(x))))
        return self.act(main + y)


class UpsamplingBottleneck(nn.Module):
    """MaxUnpool 대신 bilinear 업샘플 + 1x1 conv로 복원한다(ONNX export 가능)."""

    def __init__(self, in_ch, out_ch, internal_ratio=4, dropout=0.1):
        super().__init__()
        internal = in_ch // internal_ratio
        self.main_conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.proj = nn.Sequential(nn.Conv2d(in_ch, internal, 1, bias=False),
                                  nn.BatchNorm2d(internal), nn.PReLU(internal))
        self.conv = nn.Sequential(nn.Conv2d(internal, internal, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(internal), nn.PReLU(internal))
        self.expand = nn.Sequential(nn.Conv2d(internal, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.drop = nn.Dropout2d(dropout)
        self.act = nn.PReLU(out_ch)

    def forward(self, x):
        size = (x.shape[-2] * 2, x.shape[-1] * 2)
        main = F.interpolate(self.main_conv(x), size=size, mode="bilinear", align_corners=False)
        y = F.interpolate(self.proj(x), size=size, mode="bilinear", align_corners=False)
        y = self.drop(self.expand(self.conv(y)))
        return self.act(main + y)


class ENet(nn.Module):
    def __init__(self, num_classes=5, **kwargs):
        super().__init__()
        self.initial = InitialBlock(16)
        # Stage 1
        self.down1 = DownsamplingBottleneck(16, 64, dropout=0.01)
        self.stage1 = nn.Sequential(*[RegularBottleneck(64, dropout=0.01) for _ in range(4)])
        # Stage 2 + Stage 3 (dilated / asymmetric bottlenecks)
        self.down2 = DownsamplingBottleneck(64, 128)
        self.stage2 = self._middle()
        self.stage3 = self._middle()
        # Decoder
        self.up4 = UpsamplingBottleneck(128, 64)
        self.stage4 = nn.Sequential(RegularBottleneck(64), RegularBottleneck(64))
        self.up5 = UpsamplingBottleneck(64, 16)
        self.stage5 = RegularBottleneck(16)
        self.fullconv = nn.ConvTranspose2d(16, num_classes, 3, stride=2, padding=1, output_padding=1)

    @staticmethod
    def _middle():
        return nn.Sequential(
            RegularBottleneck(128),
            RegularBottleneck(128, dilation=2, padding=2),
            RegularBottleneck(128, kernel=5, padding=2, asymmetric=True),
            RegularBottleneck(128, dilation=4, padding=4),
            RegularBottleneck(128),
            RegularBottleneck(128, dilation=8, padding=8),
            RegularBottleneck(128, kernel=5, padding=2, asymmetric=True),
            RegularBottleneck(128, dilation=16, padding=16),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        y = self.initial(x)
        y = self.stage1(self.down1(y))
        y = self.stage2(self.down2(y))
        y = self.stage3(y)
        y = self.stage4(self.up4(y))
        y = self.stage5(self.up5(y))
        seg_logits = self.fullconv(y)
        if seg_logits.shape[-2:] != (h, w):
            seg_logits = F.interpolate(seg_logits, size=(h, w), mode="bilinear", align_corners=False)
        return {"seg_logits": seg_logits, "restored_image": x}

    def forward_export(self, x):
        outputs = self.forward(x)
        return outputs["restored_image"], outputs["seg_logits"]
