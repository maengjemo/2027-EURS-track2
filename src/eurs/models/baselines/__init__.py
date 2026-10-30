"""비교군 baseline 모델(순수 torch 자체 구현)과 학습 없는 복원 기법.

분할 baseline(enet/cgnet/fpenet/segformer_b0)은 v0/DuoLite와 같은 인터페이스
``forward(x) -> dict(seg_logits, restored_image=x)``를 가지며 복원은 Identity다.
uwcnn은 복원 전용(seg_logits는 0 텐서)이다.
"""
from .cgnet import CGNet
from .enet import ENet
from .fpenet import FPENet
from .restore_nolearn import METHODS, grayworld_clahe, identity
from .segformer_b0 import SegFormerB0
from .uwcnn import UWCNN

__all__ = ["CGNet", "ENet", "FPENet", "SegFormerB0", "UWCNN",
           "METHODS", "grayworld_clahe", "identity"]
