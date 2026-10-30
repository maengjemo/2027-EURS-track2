"""UWCNN형 소형 복원 CNN (Li et al. 2020, "Underwater scene prior inspired ...").

3개의 밀집 블록(각 블록: 3x3 conv×3, 16 filters)의 출력과 블록 입력을 concat하며 쌓고,
마지막 1x1/3x3 conv로 3채널 잔차를 만들어 입력에 더한다(약 0.04M). 복원 전용이므로
seg_logits는 0 텐서를 낸다.

인터페이스: forward(x) -> dict(restored_image=(B,3,H,W), seg_logits=zeros(B,5,H,W)).
"""
import torch
from torch import nn


class UWBlock(nn.Module):
    def __init__(self, in_ch, growth=16):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, growth, 3, padding=1)
        self.c2 = nn.Conv2d(growth, growth, 3, padding=1)
        self.c3 = nn.Conv2d(growth, growth, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y1 = self.act(self.c1(x))
        y2 = self.act(self.c2(y1))
        y3 = self.act(self.c3(y2))
        return torch.cat([y1, y2, y3, x], dim=1)


class UWCNN(nn.Module):
    def __init__(self, num_classes=5, growth=16, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.block1 = UWBlock(3, growth)
        self.block2 = UWBlock(3 + 3 * growth, growth)
        self.block3 = UWBlock(3 + 2 * 3 * growth, growth)
        self.out = nn.Conv2d(3 + 3 * 3 * growth, 3, 3, padding=1)

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        residual = self.out(y)
        restored = torch.clamp(x + residual, 0.0, 1.0)
        seg_logits = x.new_zeros((x.shape[0], self.num_classes, x.shape[-2], x.shape[-1]))
        return {"restored_image": restored, "seg_logits": seg_logits}

    def forward_export(self, x):
        outputs = self.forward(x)
        return outputs["restored_image"], outputs["seg_logits"]
