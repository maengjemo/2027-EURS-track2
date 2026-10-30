from .eurs_v0 import EURSv0, ExportModel
from .duolite_s1 import DuoLiteS1
from .duolite_pre import DuoLitePre
from .student_v2 import StudentAttnIter, StudentBiSe, StudentDSFPN
from .student_distill import StudentDistill
from .baselines import CGNet, ENet, FPENet, SegFormerB0, UWCNN

_REGISTRY = {"eurs_v0": EURSv0, "duolite_s1": DuoLiteS1, "duolite_pre": DuoLitePre,
             "enet": ENet, "cgnet": CGNet, "fpenet": FPENet,
             "segformer_b0": SegFormerB0, "uwcnn": UWCNN,
             "student_dsfpn": StudentDSFPN, "student_bise": StudentBiSe,
             "student_attniter": StudentAttnIter, "student_distill": StudentDistill}


def build_model(name="eurs_v0", **kwargs):
    """config의 model.name으로 모델을 만든다.

    등록: eurs_v0, duolite_s1(본 방법), duolite_pre(사전학습 인코더 변형),
    enet·cgnet·fpenet·segformer_b0(분할 baseline), uwcnn(복원 baseline).
    분할 baseline은 width 같은 미지원 kwargs를 조용히 무시한다.
    duolite_pre는 backbone·pretrained·dec·decoder kwargs를 받는다.
    student_dsfpn·student_bise·student_attniter는 학생 v2 설계 1·2·3이다(docs/student_v2.md).
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model name {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


__all__ = ["EURSv0", "DuoLiteS1", "DuoLitePre", "ExportModel", "build_model",
           "StudentDSFPN", "StudentBiSe", "StudentAttnIter", "StudentDistill",
           "CGNet", "ENet", "FPENet", "SegFormerB0", "UWCNN"]
