"""FPENet (Liu & Yin 2019, arXiv:1909.08599) 순수 torch 자체 구현.

Feature Pyramid Encoding(FPE) block: 1x1 확장 후 채널을 scales개로 나눠 서로 다른
dilation의 depthwise 3x3을 계층적으로 적용(Res2Net식 누적)하고 1x1로 합친다. 디코더는
MEU(Mutual Embedding Upsample): 고수준 특징에서 채널 어텐션, 저수준 특징에서 공간
어텐션을 얻어 결합한다. width=16, block_num=[1,3,9] 구성이다.

인터페이스: forward(x) -> dict(seg_logits=(B,5,H,W), restored_image=x). 복원은 Identity다.
"""
import torch
from torch import nn
from torch.nn import functional as F


class FPEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilat=(1, 2, 4, 8), stride=1, t=1, scales=4, downsample=None):
        super().__init__()
        mid = out_ch * t
        assert mid % scales == 0
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, mid, 1, bias=False), nn.BatchNorm2d(mid))
        width = mid // scales
        self.scales = scales
        self.stride = stride
        self.conv2 = nn.ModuleList([
            nn.Conv2d(width, width, 3, stride=stride, padding=dilat[i], dilation=dilat[i],
                      groups=width, bias=False) for i in range(scales)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(scales)])
        self.conv3 = nn.Sequential(nn.Conv2d(mid, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        chunks = torch.chunk(out, self.scales, dim=1)
        outs = []
        for i in range(self.scales):
            # stride==1일 때만 상위 scale 출력을 누적(Res2Net식). stride==2에서는 크기가
            # 달라지므로 누적하지 않는다.
            inp = chunks[i] if (i == 0 or self.stride != 1) else chunks[i] + outs[-1]
            outs.append(self.relu(self.bn2[i](self.conv2[i](inp))))
        out = self.conv3(torch.cat(outs, dim=1))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class MEUModule(nn.Module):
    """Mutual Embedding Upsample: 고수준=채널 어텐션, 저수준=공간 어텐션."""

    def __init__(self, channels_high, channels_low, channel_out):
        super().__init__()
        self.conv_low = nn.Sequential(nn.Conv2d(channels_low, channel_out, 1, bias=False),
                                      nn.BatchNorm2d(channel_out))
        self.sa_conv = nn.Conv2d(1, 1, 1, bias=False)
        self.conv_high = nn.Sequential(nn.Conv2d(channels_high, channel_out, 1, bias=False),
                                       nn.BatchNorm2d(channel_out))
        self.ca_conv = nn.Conv2d(channel_out, channel_out, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low):
        h, w = fms_low.shape[-2:]
        # 채널 어텐션은 conv_high로 channel_out 차원에 투영한 뒤 계산한다(ca_conv 입력이
        # channel_out이어야 하므로). 순서를 바꾸지 않으면 channels_high != channel_out인
        # MEU에서 채널 수가 불일치한다.
        fms_high = self.conv_high(fms_high)
        fms_low = self.conv_low(fms_low)
        ca = self.sigmoid(self.relu(self.ca_conv(fms_high.mean(dim=(2, 3), keepdim=True))))
        sa = self.sigmoid(self.sa_conv(fms_low.mean(dim=1, keepdim=True)))
        fms_high_up = F.interpolate(fms_high, size=(h, w), mode="bilinear", align_corners=False)
        return fms_high_up * ca + fms_low * sa


class FPENet(nn.Module):
    def __init__(self, num_classes=5, base_width=16, scales=4, **kwargs):
        super().__init__()
        base = int(base_width)
        outplanes = [base, base * 2, base * 4]
        block_num = [1, 3, 9]
        dilat = (1, 2, 4, 8)
        self.conv1 = nn.Sequential(nn.Conv2d(3, outplanes[0], 3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(outplanes[0]), nn.ReLU(inplace=True))
        self.inplanes = outplanes[0]
        self.layer1 = self._make_layer(outplanes[0], block_num[0], dilat, stride=1, t=1, scales=scales)
        self.layer2 = self._make_layer(outplanes[1], block_num[1], dilat, stride=2, t=4, scales=scales)
        self.layer3 = self._make_layer(outplanes[2], block_num[2], dilat, stride=2, t=4, scales=scales)
        self.meu1 = MEUModule(outplanes[2], outplanes[1], outplanes[2])
        self.meu2 = MEUModule(outplanes[2], outplanes[0], outplanes[1])
        self.project = nn.Conv2d(outplanes[1], num_classes, 1)

    def _make_layer(self, planes, blocks, dilat, stride, t, scales):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes))
        layers = [FPEBlock(self.inplanes, planes, dilat, stride=stride, t=t, scales=scales, downsample=downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(FPEBlock(planes, planes, dilat, stride=1, t=t, scales=scales))
        return nn.Sequential(*layers)

    def forward(self, x):
        h, w = x.shape[-2:]
        y = self.conv1(x)
        x1 = self.layer1(y)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x2 = self.meu1(x3, x2)
        x1 = self.meu2(x2, x1)
        seg_logits = F.interpolate(self.project(x1), size=(h, w), mode="bilinear", align_corners=False)
        return {"seg_logits": seg_logits, "restored_image": x}

    def forward_export(self, x):
        outputs = self.forward(x)
        return outputs["restored_image"], outputs["seg_logits"]
