"""DuoLite-S1 student model for the EURS Track 2 budget.

이 파일은 아키텍처 결정 문서(안 B, DuoLite-S1)가 확정하고 참조 구현
``arch_probe/model.py``(``EURSStudent``, StudentConfig 기본값)가 검증한 학생 모델을
구조적으로 동일하게 옮긴 것이다. 이전 버전은 참조 구현을 읽지 못해 PPM 대신 SE
전역 문맥을 쓰는 형태로 재구성했으나(0.8173M / 3.426 GFLOPs), 이 버전은 참조 구현의
블록·채널·스테이지 깊이·전역 보정을 그대로 복제한다.

설계 규칙(참조 구현 docstring 그대로):
  * 사용 op: Conv, BatchNorm(export 시 fuse), Relu, Clip, Sigmoid, Add, Mul, Sub,
    Concat, Resize(bilinear, half_pixel), Reshape, MatMul(GAP), Slice. LN/GN/GELU/
    Softmax/FFT/grid_sample/AvgPool/ReduceMean은 쓰지 않는다.
  * 전역 평균 풀링(GAP)은 AveragePool 계열 대신 Reshape + 상수 MatMul로 구현한다
    (``gap_matmul``). export가 정적 입력이므로 1/(H*W) 상수가 그래프에 그대로 굽혀
    Reciprocal 없이 MatMul 하나로 남는다.
  * 파라미터는 대부분 stride 16/32(stage3/4)에 두고 FLOPs를 낮춰 같은 가중치를
    384x768 / 512x1024 / 576x1152로 export할 수 있게 한다.
  * 분할 디코더와 복원 분기는 인코더만 공유한다(복원↔분할 직접 결합 없음).
  * presence head(R1): 이미지별 클래스 존재 로짓 5개를 낸다. 추론·export 시
    seg_logits += class_bias + presence_gain * sigmoid(presence). class_bias·
    presence_gain은 학습 후 캘리브레이션되어 initializer로 굽힌다.
  * medium head: water-type 10-way + depth 회귀(보조 손실). FiLM으로 stride 8
    디코더 특징을 조건화한다.
  * 복원(R4): 문맥 벡터에서 전역 구조 보정(gain/bias, 3x3 ΔCCM, 단조 조각선형
    톤커브)을 예측하고, stride 2 저해상도 잔차와 작은 풀해상도 refiner를 더한다.

인터페이스는 패키지 규약을 따른다:
  * ``forward(x) -> dict(restored_image, seg_logits, presence_logits,
    water_logits, depth)``
  * ``forward_export(x) -> (restored_image, seg_logits)``
  * ``class_bias`` 파라미터(캘리브레이션 로짓 바이어스 자리; tools/calibrate_bias.py,
    eurs/export/onnx_export.py:apply_calibration이 참조).
  * ``width`` 인자(채널 폭 스케일; ``build_model("duolite_s1", width=...)``).

참조 구현과의 의도적 차이(구조·파라미터 수·FLOPs는 불변):
  * ``class_bias``는 (5,) 형태로 둔다. 참조 구현은 (1,5,1,1)이지만 apply_calibration이
    R^5 리스트를 copy_로 굽으므로 (5,)라야 브로드캐스트 없이 그대로 복사된다.
    forward에서 reshape(1,5,1,1)로 더한다.
  * 참조 구현의 medium head(Conv2d ctx_hidden->n_water+1=11)를 그대로 두고 출력만
    water_logits(앞 10)·depth(마지막 1)로 분리한다(파라미터 수 동일).

예산(참조 목표: 0.8809M params, 3.438 GFLOPs @384x768, 6.112 @512x1024).
이 모듈은 참조 구현 ``EURSStudent``(StudentConfig 기본값)와 블록·채널·깊이·헤드가
동일하다. **실측(2026-09-30, 메인 세션, third_party/EURS_check_submission_flops.py)**:
PyTorch 896,788 params; ONNX 0.8806M initializers, 3.4378 GFLOPs @384x768,
6.1116 GFLOPs @512x1024, PASSED, 어휘 밖 op 경고 없음, onnxruntime 대조 오차 0.0.
tests/test_duolite.py의 ``test_duolite_export_and_organizer_budget``가 캡 준수를 강제한다.

참고: GAP를 상수 MatMul(``gap_matmul``)로 구현하므로, constant-folding으로 1/(H*W)
상수 벡터가 initializer가 되면 export params가 참조 목표보다 해상도의 H*W(1/32 특징
크기)만큼 미세하게 커질 수 있다. 캡(1.0M) 안이므로 문제되지 않으며, 차이가 크면
이 자리에 원인을 실측과 함께 적는다.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import channels


# --------------------------------------------------------------------------- #
# building blocks (참조 구현 arch_probe/model.py 그대로)
# --------------------------------------------------------------------------- #
class ConvBNAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, groups: int = 1,
                 act: bool = True):
        layers = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
                  nn.BatchNorm2d(cout)]
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class InvertedResidual(nn.Module):
    """MobileNetV2식 블록: 1x1 expand -> DW3x3 -> 1x1 project (+residual)."""

    def __init__(self, cin: int, cout: int, stride: int = 1, expand: float = 3.0):
        super().__init__()
        hidden = int(round(cin * expand))
        self.use_res = stride == 1 and cin == cout
        self.expand = ConvBNAct(cin, hidden, 1)
        self.dw = ConvBNAct(hidden, hidden, 3, stride, groups=hidden)
        self.project = ConvBNAct(hidden, cout, 1, act=False)

    def forward(self, x):
        y = self.project(self.dw(self.expand(x)))
        return x + y if self.use_res else y


class DWSep(nn.Module):
    """Depthwise-separable refinement: DW3x3 -> 1x1, 둘 다 BN+ReLU."""

    def __init__(self, c: int):
        super().__init__()
        self.dw = ConvBNAct(c, c, 3, groups=c)
        self.pw = ConvBNAct(c, c, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


def gap_matmul(x: torch.Tensor) -> torch.Tensor:
    """전역 평균 풀링을 Reshape+상수 MatMul로 구현한다(vocabulary-safe, 정적 shape).

    정적 export에서 H*W가 파이썬 상수이므로 1/(H*W)가 그대로 상수 벡터로 굽혀
    Reciprocal 없이 MatMul 하나로 남는다.
    """
    n, c, h, w = x.shape
    flat = x.reshape(n, c, h * w)
    avg = torch.full((h * w, 1), 1.0 / float(h * w), dtype=x.dtype, device=x.device)
    return torch.matmul(flat, avg).reshape(n, c, 1, 1)


class GlobalContext(nn.Module):
    """전역 벡터에서 채널 게이트를 만드는 SE식 블록(Sigmoid + Mul)."""

    def __init__(self, c: int, r: int = 4):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c // r, 1)
        self.fc2 = nn.Conv2d(c // r, c, 1)

    def forward(self, x, g):
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(g))))
        return x * s


def up2(x: torch.Tensor) -> torch.Tensor:
    # Resize(linear, half_pixel), 상수 scales -> TensorRT bilinear
    return F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)


# --------------------------------------------------------------------------- #
# config (참조 구현 StudentConfig 기본값)
# --------------------------------------------------------------------------- #
@dataclass
class StudentConfig:
    stem: int = 24
    widths: Tuple[int, int, int, int] = (32, 64, 128, 160)   # strides 4, 8, 16, 32
    depths: Tuple[int, int, int, int] = (1, 2, 4, 2)         # s2 블록 뒤 반복 수
    expands: Tuple[float, float, float, float] = (2.0, 3.0, 3.0, 2.0)
    dec: int = 64            # FPN-lite 채널
    ctx_hidden: int = 64     # 문맥 헤드 공유 hidden
    n_classes: int = 5
    n_water: int = 10
    tone_knots: int = 8      # 채널당 조각선형 톤커브 knot 수
    res_local: int = 16      # 저해상도(stride 2) 복원 잔차 채널
    refiner: int = 8         # 풀해상도 refiner 폭
    film: bool = True
    decoder_full_conv: bool = False  # stride 4 3x3 full conv 옵션(R6 knob)


def _scaled_config(width: float) -> StudentConfig:
    """width로 채널 폭을 스케일한다. width=1.0이면 참조 기본값 그대로."""
    base = StudentConfig()
    if width == 1.0:
        return base
    w = lambda v: channels(v, width)
    return replace(
        base,
        stem=w(base.stem),
        widths=tuple(w(v) for v in base.widths),
        dec=w(base.dec),
        ctx_hidden=w(base.ctx_hidden),
        res_local=w(base.res_local),
        refiner=w(base.refiner),
    )


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class DuoLiteS1(nn.Module):
    """공유 인코더 + FPN-lite 분할 디코더 + 전역/로컬/refiner 복원 분기.

    forward(x) -> dict(restored_image, seg_logits, presence_logits(B,5),
                       water_logits(B,10), depth(B,1))
    forward_export(x) -> (restored_image, seg_logits)
    """

    def __init__(self, width: float = 1.0):
        super().__init__()
        self.cfg = cfg = _scaled_config(float(width))
        c1, c2, c3, c4 = cfg.widths
        e1, e2, e3, e4 = cfg.expands
        d1, d2, d3, d4 = cfg.depths

        # ---- encoder ---------------------------------------------------- #
        self.stem = ConvBNAct(3, cfg.stem, 3, 2)                              # H/2
        self.stage1 = nn.Sequential(
            InvertedResidual(cfg.stem, c1, 2, e1),
            *[InvertedResidual(c1, c1, 1, e1) for _ in range(d1)])            # H/4
        self.stage2 = nn.Sequential(
            InvertedResidual(c1, c2, 2, e2),
            *[InvertedResidual(c2, c2, 1, e2) for _ in range(d2)])            # H/8
        self.stage3 = nn.Sequential(
            InvertedResidual(c2, c3, 2, e3),
            *[InvertedResidual(c3, c3, 1, e3) for _ in range(d3)])            # H/16
        self.stage4 = nn.Sequential(
            InvertedResidual(c3, c4, 2, e4),
            *[InvertedResidual(c4, c4, 1, e4) for _ in range(d4)])            # H/32
        self.gctx = GlobalContext(c4)

        # ---- context heads (image-level) -------------------------------- #
        self.ctx_fc = nn.Sequential(nn.Conv2d(c4, cfg.ctx_hidden, 1), nn.ReLU(inplace=True))
        self.presence_head = nn.Conv2d(cfg.ctx_hidden, cfg.n_classes, 1)      # R1
        self.medium_head = nn.Conv2d(cfg.ctx_hidden, cfg.n_water + 1, 1)      # water 10 + depth 1
        n_glob = 3 + 3 + 9 + 3 * cfg.tone_knots                              # gain, bias, CCM, tone slopes
        self.restore_global = nn.Conv2d(cfg.ctx_hidden, n_glob, 1)
        nn.init.zeros_(self.restore_global.weight); nn.init.zeros_(self.restore_global.bias)
        if cfg.film:
            self.film = nn.Conv2d(cfg.ctx_hidden, 2 * cfg.dec, 1)
            nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

        # ---- segmentation decoder (FPN-lite) ---------------------------- #
        d = cfg.dec
        self.lat4 = ConvBNAct(c4, d, 1, act=False)
        self.lat3 = ConvBNAct(c3, d, 1, act=False)
        self.lat2 = ConvBNAct(c2, d, 1, act=False)
        self.lat1 = ConvBNAct(c1, d, 1, act=False)
        self.ref3 = DWSep(d)
        self.ref2 = DWSep(d)
        self.ref1 = ConvBNAct(d, d, 3) if cfg.decoder_full_conv else DWSep(d)
        self.seg_head = nn.Conv2d(d, cfg.n_classes, 1)
        # coverage 방어 벡터(학습 후 캘리브레이션되어 initializer로 굽힘).
        # class_bias는 (5,): apply_calibration이 R^5 리스트를 copy_로 굽으므로 브로드캐스트 없이 복사된다.
        self.class_bias = nn.Parameter(torch.zeros(cfg.n_classes))
        self.presence_gain = nn.Parameter(torch.zeros(1, cfg.n_classes, 1, 1))

        # ---- restoration branch ----------------------------------------- #
        r = cfg.res_local
        self.res_lat = ConvBNAct(c1, r, 1)                                    # stage1(H/4)에서
        self.res_mix = nn.Sequential(ConvBNAct(r + cfg.stem, r + cfg.stem, 3, groups=r + cfg.stem),
                                     ConvBNAct(r + cfg.stem, r, 1))           # H/2
        self.res_out = nn.Conv2d(r, 3, 1)
        nn.init.zeros_(self.res_out.weight); nn.init.zeros_(self.res_out.bias)
        self.refiner = nn.Sequential(nn.Conv2d(6, cfg.refiner, 3, 1, 1), nn.ReLU(inplace=True),
                                     nn.Conv2d(cfg.refiner, 3, 3, 1, 1))      # full res
        nn.init.zeros_(self.refiner[2].weight); nn.init.zeros_(self.refiner[2].bias)

    # ------------------------------------------------------------------ #
    def _global_correct(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """구조화된 전역 색 보정. 예측이 0이면 항등(identity).
        x: (B,3,H,W) in [0,1]; p: (B,n_glob,1,1)."""
        K = self.cfg.tone_knots
        gain = 1.0 + p[:, 0:3]
        bias = p[:, 3:6]
        ccm = p[:, 6:15]                       # 3x3 행렬의 delta, row-major
        slopes = 2.0 * torch.sigmoid(p[:, 15:15 + 3 * K]) - 1.0   # (-1,1) -> 단조
        y = x * gain + bias
        # 색 보정 행렬 M = I + dM, Slice/Mul/Add로 적용
        r, g, b = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        r2 = r * (1.0 + ccm[:, 0:1]) + g * ccm[:, 1:2] + b * ccm[:, 2:3]
        g2 = r * ccm[:, 3:4] + g * (1.0 + ccm[:, 4:5]) + b * ccm[:, 5:6]
        b2 = r * ccm[:, 6:7] + g * ccm[:, 7:8] + b * (1.0 + ccm[:, 8:9])
        y = torch.cat([r2, g2, b2], dim=1)
        # 채널당 단조 조각선형 톤커브 on [0,1]
        y = torch.clamp(y, 0.0, 1.0)
        out = y
        seg_w = 1.0 / K
        for k in range(K):
            s_k = slopes[:, 3 * k:3 * k + 3]
            out = out + s_k * torch.clamp(y - k * seg_w, 0.0, seg_w)
        return out

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        s0 = self.stem(x)            # H/2
        s1 = self.stage1(s0)         # H/4
        s2 = self.stage2(s1)         # H/8
        s3 = self.stage3(s2)         # H/16
        s4 = self.stage4(s3)         # H/32
        g = gap_matmul(s4)
        s4 = self.gctx(s4, g)
        g = gap_matmul(s4)

        h = self.ctx_fc(g)
        presence = self.presence_head(h)          # (B,5,1,1)
        medium = self.medium_head(h)              # (B,11,1,1)
        glob = self.restore_global(h)

        # ---- segmentation ------------------------------------------- #
        d = cfg.dec
        p4 = self.lat4(s4)
        p3 = self.ref3(self.lat3(s3) + up2(p4))
        p2 = self.lat2(s2) + up2(p3)
        if cfg.film:
            fb = self.film(h)
            p2 = p2 * (1.0 + fb[:, :d]) + fb[:, d:]
        p2 = self.ref2(p2)
        p1 = self.ref1(self.lat1(s1) + up2(p2))
        logits = self.seg_head(p1)
        logits = logits + self.class_bias.reshape(1, cfg.n_classes, 1, 1) \
            + self.presence_gain * torch.sigmoid(presence)
        seg_logits = F.interpolate(logits, scale_factor=4.0, mode="bilinear", align_corners=False)

        # ---- restoration -------------------------------------------- #
        base = self._global_correct(x, glob)
        loc = self.res_mix(torch.cat([up2(self.res_lat(s1)), s0], dim=1))
        resid = F.interpolate(self.res_out(loc), scale_factor=2.0, mode="bilinear", align_corners=False)
        y = base + resid
        y = y + self.refiner(torch.cat([x, y], dim=1))
        restored = torch.clamp(y, 0.0, 1.0)

        return {"restored_image": restored, "seg_logits": seg_logits,
                "presence_logits": presence.flatten(1),
                "water_logits": medium[:, :cfg.n_water].flatten(1),
                "depth": medium[:, cfg.n_water:].flatten(1)}

    def forward_export(self, x: torch.Tensor):
        outputs = self.forward(x)
        return outputs["restored_image"], outputs["seg_logits"]
