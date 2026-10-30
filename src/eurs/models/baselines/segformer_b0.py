"""SegFormer-B0 분할 baseline: teacher_seg.TeacherSeg를 mit-b0 backbone으로 재사용.

Track 1 참조(>1M, ImageNet 사전학습 효과 비교)용이다. TeacherSeg는 seg_logits와
presence_logits를 내므로 여기에 restored_image=x(Identity)를 붙여 v0/DuoLite와 같은
인터페이스로 맞춘다. transformers 의존은 TeacherSeg 생성자에서 지연 import된다.
"""
from torch import nn


class SegFormerB0(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, **kwargs):
        super().__init__()
        from eurs.models.teacher_seg import TeacherSeg
        self.seg = TeacherSeg(backbone="nvidia/mit-b0", num_labels=num_classes, pretrained=pretrained)

    def forward(self, x):
        out = self.seg(x)
        return {"seg_logits": out["seg_logits"], "restored_image": x,
                "presence_logits": out["presence_logits"]}

    def forward_export(self, x):
        outputs = self.forward(x)
        return outputs["restored_image"], outputs["seg_logits"]
