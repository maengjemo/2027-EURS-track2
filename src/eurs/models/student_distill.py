"""학생 v3(SM9): DS-FPN 학생 + 증류 헤드 + DINOv3 결합 교사 특징 추출기 (docs/WORK_ORDER_encoder_predistill_b200.md).

- Stage A(인코더 사전 증류): ``StudentDistill.encode``의 인코더 특징을 ``distill_heads``(학습 전용 1x1 투영)로
  교사 토큰 공간에 올려 ``TeacherFeatures``의 토큰과 cosine 정렬한다(손실은 eurs/losses/distill.py).
- Stage B(본 학습): 같은 학생을 ``load_encoder``로 시작해 GT + 외부 의사 라벨 + 온라인 로짓 KD.
- 배포 그래프(``forward_export``)는 StudentDSFPN과 같고 distill_heads를 지나지 않는다(ONNX params에 안 들어감).

교사 특징 추출은 ``TeacherDinoLoRA.forward``의 전처리를 그대로 따른다: replicate 패딩(patch 배수) →
ImageNet 정규화 → ``_adapter_grid`` 선설정(DeBA hook이 읽는다) → ``output_hidden_states``. 실측(2026-10-26,
T1_seg_dinov3l_deba_dpt_meta_ext5_bf16e20): 24층·hidden 1024·patch 16, prefix 토큰 5(CLS 1 + register 4),
``last_hidden_state = encoder.norm(hidden_states[-1])``, DeBA는 중간층 hidden_states에도 반영된다.
"""
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .student_v2 import StudentDSFPN

TOKEN_DIM_KEY = "teacher_hidden"
STUDENT_LEVELS = ("s1", "s2", "s3", "s4")          # stride 4 / 8 / 16 / 32
DEFAULT_DISTILL = ({"student": "s3", "teacher": "mid12"},
                   {"student": "s4", "teacher": "final"})


def teacher_tap_channels(tap: str, hidden: int, head_ch: int) -> int:
    """교사 탭 이름 → 채널 수. mid<k>·final = ViT hidden, head16·head8 = DPT 헤드 채널."""
    if tap == "final" or (tap.startswith("mid") and tap[3:].isdigit()):
        return int(hidden)
    if tap in ("head16", "head8"):
        return int(head_ch)
    raise ValueError(f"Unknown teacher tap {tap!r} (mid<k> | final | head16 | head8)")


class TeacherFeatures(nn.Module):
    """동결 결합 교사(TeacherDinoLoRA, DeBA + DPT + 메타 FiLM)에서 지정한 탭의 특징 지도를 뽑는다.

    forward(x, meta) -> {"mid12": (B,1024,H/16,W/16), "final": (B,1024,H/16,W/16),
                         "head16": (B,256,H/16,W/16), "head8": (B,256,H/8,W/8), "seg_logits": (B,5,H,W)}
    (요청한 탭만 계산; ``with_logits``면 seg_logits도). 교사는 eval·no_grad이고 bf16 autocast는 호출자가 건다.
    ``meta``(B,11)는 교사가 ``meta_cond``일 때 head 탭·로짓에 필요하다(토큰 탭에는 필요 없음).
    """

    def __init__(self, teacher: nn.Module, taps: Sequence[str] = ("mid12", "final"), with_logits: bool = True):
        super().__init__()
        if getattr(teacher, "head_name", None) != "dpt" or not hasattr(teacher, "encoder"):
            raise ValueError("TeacherFeatures expects a TeacherDinoLoRA with a DPT head")
        self.teacher = teacher.eval()
        self.teacher.requires_grad_(False)
        self.taps = tuple(taps)
        self.with_logits = bool(with_logits)
        n_layers = len(teacher.encoder.model.layer)
        for tap in self.taps:
            teacher_tap_channels(tap, 1, 1)  # 이름 검사
            if tap.startswith("mid") and not 1 <= int(tap[3:]) <= n_layers:
                raise ValueError(f"{tap}: layer must be 1..{n_layers}")
        self.hidden = int(teacher.encoder.config.hidden_size)
        self.head_ch = int(teacher.head_ch)

    def train(self, mode: bool = True):
        super().train(False)  # 항상 eval
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor, meta: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        t = self.teacher
        h, w = x.shape[-2:]
        p = t.patch_size
        padded = F.pad(x, (0, (-w) % p, 0, (-h) % p), mode="replicate")
        ph, pw = padded.shape[-2:]
        gh, gw = ph // p, pw // p
        t._adapter_grid = (gh, gw)
        need_hidden = any(tap.startswith("mid") for tap in self.taps)
        out = t.encoder((padded - t.image_mean) / t.image_std, output_hidden_states=need_hidden, return_dict=True)
        tokens = out.last_hidden_state
        prefix = tokens.shape[1] - gh * gw
        if prefix < 1:
            raise ValueError("DINO output token count does not match the patch grid")

        def grid(tok):
            return tok[:, prefix:].transpose(1, 2).reshape(x.shape[0], -1, gh, gw)

        feats = {}
        for tap in self.taps:
            if tap.startswith("mid"):
                feats[tap] = grid(out.hidden_states[int(tap[3:])])
            elif tap == "final":
                feats[tap] = grid(tokens)
        need_head = self.with_logits or any(tap.startswith("head") for tap in self.taps)
        if need_head:
            if t.meta_cond and (meta is None or meta.shape != (x.shape[0], 11)):
                raise ValueError("meta_cond teacher needs a (B,11) metadata tensor for head taps / logits")
            head = t.head
            f16 = head.project(grid(tokens))
            if t.meta_cond:
                f16 = t._apply_film(f16, meta)
            if "head16" in self.taps:
                feats["head16"] = f16
            f8 = head.up1(F.interpolate(f16, scale_factor=2, mode="bilinear", align_corners=False))
            if "head8" in self.taps:
                feats["head8"] = f8
            if self.with_logits:
                f4 = head.up2(F.interpolate(f8, scale_factor=2, mode="bilinear", align_corners=False))
                logits = F.interpolate(head.classifier(f4), size=(ph, pw), mode="bilinear", align_corners=False)
                feats["seg_logits"] = logits[..., :h, :w]
        return feats


class DistillHeads(nn.Module):
    """학습 전용 투영기 묶음: 학생 특징(c_s) → 교사 탭 채널(c_t), 1x1 → BN → ReLU → 1x1.

    key = "<student>_<teacher>" (예: s3_mid12). forward는 투영 결과를 같은 key로 돌려준다.
    """

    def __init__(self, spec: Iterable[dict], student_channels: Dict[str, int], hidden: int, head_ch: int,
                 proj_hidden: Optional[int] = None):
        super().__init__()
        self.spec: List[dict] = []
        self.proj = nn.ModuleDict()
        for item in spec:
            s, t = str(item["student"]), str(item["teacher"])
            if s not in student_channels:
                raise ValueError(f"Unknown student level {s!r}; expected one of {sorted(student_channels)}")
            cin, cout = int(student_channels[s]), teacher_tap_channels(t, hidden, head_ch)
            mid = int(proj_hidden) if proj_hidden else max(cin, 256)
            key = f"{s}_{t}"
            if key in self.proj:
                raise ValueError(f"Duplicate distill pair {key}")
            self.proj[key] = nn.Sequential(nn.Conv2d(cin, mid, 1, bias=False), nn.BatchNorm2d(mid),
                                           nn.ReLU(inplace=True), nn.Conv2d(mid, cout, 1))
            self.spec.append({"student": s, "teacher": t, "key": key})

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {item["key"]: self.proj[item["key"]](feats[item["student"]]) for item in self.spec}

    def teacher_taps(self) -> List[str]:
        seen = []
        for item in self.spec:
            if item["teacher"] not in seen:
                seen.append(item["teacher"])
        return seen


class StudentDistill(StudentDSFPN):
    """DS-FPN 학생(설계 1) + distill_heads. 기본 인코더 lcnet_075(stride 4/8/16/32 = 48/96/192/384ch).

    예산(strict export 실측, 640x768, 투영기 제외): dec (64,48,32,24) 0.820M/4.03 GF, (96,64,48,32) 0.875M/4.98 GF
    (기본), (128,96,64,48) 0.954M/6.60 GF. 투영기(약 0.86M)는 학습 전용이라 ONNX params에 들어가지 않는다.

    ``distill``: [{"student": "s3", "teacher": "mid12"}, {"student": "s4", "teacher": "final"}, ...]
    교사 탭이 학생보다 세밀하면(s4 ↔ stride 16 토큰) 손실 쪽에서 교사 지도를 평균 풀링해 맞춘다.
    ``teacher_hidden``·``teacher_head_ch``는 교사 없이(예: export의 load_model) 투영기 크기를 정하기 위한 값이다.
    """

    def __init__(self, backbone: str = "lcnet_075", pretrained: bool = True, width: float = 1.0,
                 dec_widths: Sequence[int] = (96, 64, 48, 32), ppm_ch: int = 32,
                 ppm_bins: Sequence[int] = (1, 2, 3, 6), distill: Sequence[dict] = DEFAULT_DISTILL,
                 teacher_hidden: int = 1024, teacher_head_ch: int = 256, proj_hidden: Optional[int] = None,
                 refiner_ch: int = 12, refiner_depth: int = 1):
        super().__init__(backbone, pretrained, width, dec_widths, ppm_ch, ppm_bins,
                         refiner_ch=refiner_ch, refiner_depth=refiner_depth)
        channels = dict(zip(STUDENT_LEVELS, self._enc_ch))
        self.distill_heads = DistillHeads(distill, channels, teacher_hidden, teacher_head_ch, proj_hidden)

    # ---- Stage A -------------------------------------------------------- #
    def encode_dict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return dict(zip(STUDENT_LEVELS, self.encode(x)))

    def distill_forward(self, x: torch.Tensor):
        """(학생 인코더 특징 dict, 투영 dict). 디코더·헤드는 계산하지 않는다."""
        feats = self.encode_dict(x)
        return feats, self.distill_heads(feats)

    def stage_a_parameters(self):
        return list(self.encoder_parameters()) + list(self.distill_heads.parameters())

    # ---- Stage B / 평가 ------------------------------------------------- #
    def forward(self, x: torch.Tensor, return_feats: bool = False) -> Dict[str, torch.Tensor]:
        xb = self._blurred(x)                      # 추론 시 입력 저역통과 옵션(set_input_blur), 기본 항등
        s1, s2, s3, s4 = self.encode(xb)
        raw = {"s1": s1, "s2": s2, "s3": s3, "s4": s4}
        s4, h, presence, medium, glob = self.context(s4)
        f2 = self.decode(self.pre_stem(xb), s1, s2, s3, s4, h)
        seg = self._seg_out(f2, presence, x.shape[-2:])
        from .student_v2 import global_correct
        base = global_correct(x, glob, self.cfg.tone_knots)
        output = self.pack(self.restore_full(x, base, f2), seg, presence, medium)
        if return_feats:
            output["feats"] = raw
            output["distill"] = self.distill_heads(raw)
        return output

    # ---- 인코더 가중치 주고받기 ------------------------------------------ #
    ENCODER_PREFIXES = ("enc_stem.", "enc_blocks.")

    def encoder_state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if k.startswith(self.ENCODER_PREFIXES)}

    def load_encoder(self, state: Dict[str, torch.Tensor]) -> None:
        """Stage A가 저장한 인코더 가중치를 올린다. 키가 하나라도 다르면 실패(다른 백본·탭 구성 방지)."""
        expected = {k for k in self.state_dict() if k.startswith(self.ENCODER_PREFIXES)}
        given = set(state)
        if given != expected:
            missing, extra = sorted(expected - given)[:5], sorted(given - expected)[:5]
            raise ValueError(f"encoder state keys mismatch: missing {missing} extra {extra}")
        self.load_state_dict(state, strict=False)
