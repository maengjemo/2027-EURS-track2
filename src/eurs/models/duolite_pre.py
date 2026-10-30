"""DuoLite-Pre: timm features-only 백본과 교체 가능한 분할 디코더.

동기(lab-plan EURS-260930-09): DuoLite-S1은 인코더를 처음부터 학습하지만, EURS 학습셋이
작고 열화가 심해 ImageNet 사전학습 인코더가 초기 특징을 더 잘 잡을 수 있다.
기본 ``fpn_lite``는 분할 디코더·presence/medium/restore 헤드를 DuoLite-S1과 동일하게
두고 **인코더만** timm 백본으로 교체한다. EURS-261002-02는 고정 mnv2_050 인코더에서
분할 디코더만 ``fpn_wide``·``fpe``·``enet_dec``로 교체한다.

구조:
  * 백본은 ``timm.create_model(backbone, features_only=True, out_indices=(1,2,3,4))``로
    stride 4/8/16/32 특징(s1/s2/s3/s4)을 받는다. 채널 수는 ``feature_info.channels()``로
    읽어 lateral 1x1·GlobalContext·context 헤드에 동적으로 연결한다.
  * 백본은 stride 2 특징을 주지 않으므로(out_indices에 0이 없음) 복원 로컬 분기가 쓰는
    H/2 특징(``s0``)은 작은 자체 stem(``pre_stem``, Conv3x3 stride2)으로 만든다.
    기본 ``fpn_lite``에서 이것이 DuoLite-S1과의 유일한 구조적 추가다.
  * 기본 디코더(FPN-lite lat1~4 + ref1~3 + seg_head), context 헤드(ctx_fc/presence/medium/
    restore_global/FiLM), 전역 색 보정·로컬 잔차·refiner 복원 경로, GAP 상수 MatMul,
    ``class_bias``·``presence_gain``은 DuoLite-S1과 동일한 블록·수식을 쓴다
    (``duolite_s1``에서 블록·GAP·up2·StudentConfig를 import).
  * ``fpn_wide``는 128ch lateral/정제와 stride-4 DWSep 두 단을 쓴다.
    ``fpe``는 각 스테이지의 1x1 축소 + 병렬 dilated DW(1/2/4/8) + 1x1 투영,
    깊은 특징의 GAP 채널 게이트와 bilinear add로 stride 4까지 융합한다.
    ``enet_dec``는 stride 32/16/8/4에서 1x1-3x3-1x1 잔차 bottleneck을
    두 개씩 쓰며 bilinear 업샘플과 encoder skip add를 쓴다.

인터페이스는 DuoLite-S1과 같다:
  * ``forward(x) -> dict(restored_image, seg_logits, presence_logits, water_logits, depth)``
  * ``forward_export(x) -> (restored_image, seg_logits)``
  * ``class_bias`` 파라미터(캘리브레이션 로짓 바이어스 자리; apply_calibration이 참조).
  * ``encoder_parameters()``: 사전학습 백본 파라미터(train.py가 낮은 lr 그룹으로 묶음).

예산 맞춤:
  * 기본 ``fpn_lite``의 총 파라미터가 1.0M을 넘으면 디코더/FiLM 채널(기본 64)을
    후보 [64,56,48,40,32,24]
    순으로 줄여 캡 아래(안전 여유 0.985·1.0M)로 맞춘다. 실제 쓰인 값은 ``self.dec_used``.
  * 세 교체 변형의 기본 폭은 mnv2_050에서 0.95~1.0M 목표로 고정한다.
    ``dec=`` 를 명시하면 기본 폭/자동 축소를 끄고 그 값을 쓴다.
  * ``restore_head="cell_affine"``는 stride 32/16 디코더 특징에서 RGB별 gain/offset
    6채널을 예측해 입력 크기로 보간한다. 영 초기화된 1x1 투영은 항등에서 시작한다.

EURS-261002-02 실측 (mobilenetv2_050, pretrained=False 랜덤 초기화, 정적 ONNX,
``third_party/EURS_check_submission_flops.py --check_ops``; ONNX initializer params):
  decoder    PyTorch params  ONNX params  GFLOPs@384x768  GFLOPs@512x1024
  fpn_wide   980,804         0.960377M    3.489594        6.203657
  fpe        992,788         0.973273M    2.117022        3.763483
  enet_dec   978,328         0.956374M    1.904916        3.386435
세 변형 모두 주최측 cap PASSED, 어휘 밖 ONNX op 없음. 기존 백본별 기본
``fpn_lite`` 측정은 README의 "DuoLite-Pre" 표를 참조한다.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .duolite_s1 import (ConvBNAct, DWSep, GlobalContext, StudentConfig,
                         _scaled_config, gap_matmul, up2)

# 예산 캡(third_party/EURS_check_submission_flops.py DEFAULT_MAX_PARAMS_M와 동일).
_PARAM_CAP = 1_000_000
_PARAM_SAFETY = 0.985  # 자동 축소 시 목표(캡의 98.5%): export 상수 folding 여유.
_DEC_CANDIDATES = (64, 56, 48, 40, 32, 24)

# 지원 백본 후보(README 표 참조). ReLU/ReLU6 계열만 Track 2 안전 op로 export된다.
SUPPORTED_BACKBONES = (
    "tf_mobilenetv3_small_minimal_100",  # ReLU만, SE 없음(TF SAME 패딩 -> export시 Pad/Ceil)
    "mobilenetv2_050",                   # ReLU6 -> Clip
    "mobilenetv3_small_075",             # hardswish + SE (export op 위험, README 표)
    "lcnet_050",                         # hardswish
)


class _FPEEncoding(nn.Module):
    """Four parallel dilated depthwise scales, projected back to decoder width."""

    def __init__(self, in_ch: int, width: int):
        super().__init__()
        inner = max(8, in_ch // 2)
        self.reduce = ConvBNAct(in_ch, inner, 1)
        self.scales = nn.ModuleList(
            nn.Sequential(nn.Conv2d(inner, inner, 3, padding=d, dilation=d,
                                    groups=inner, bias=False),
                          nn.BatchNorm2d(inner), nn.ReLU(inplace=True))
            for d in (1, 2, 4, 8))
        self.project = ConvBNAct(4 * inner, width, 1)

    def forward(self, x):
        y = self.reduce(x)
        return self.project(torch.cat([scale(y) for scale in self.scales], dim=1))


class _MEAFusion(nn.Module):
    """Deep global channel gate followed by bilinear upsample and skip add."""

    def __init__(self, width: int):
        super().__init__()
        self.channel = nn.Conv2d(width, width, 1)

    def forward(self, deep, shallow):
        gate = torch.sigmoid(self.channel(gap_matmul(deep)))
        return shallow + F.interpolate(deep * gate, size=shallow.shape[-2:],
                                       mode="bilinear", align_corners=False)


class _ENetBottleneck(nn.Module):
    """Residual 1x1 reduction, regular 3x3, 1x1 expansion (ReLU variant)."""

    def __init__(self, width: int, extra_in: int = 0):
        super().__init__()
        inner = width // 4
        self.width = width
        self.extra_in = extra_in
        self.body = nn.Sequential(ConvBNAct(width + extra_in, inner, 1),
                                  ConvBNAct(inner, inner, 3),
                                  ConvBNAct(inner, width, 1, act=False))
        if extra_in:
            # Preserve the ordinary p1 residual and start the new input near zero.
            nn.init.normal_(self.body[0][0].weight[:, width:], std=1e-2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x if self.extra_in == 0 else x[:, :self.width]
        return self.act(residual + self.body(x))


class _AffineCorrector(nn.Module):
    """Predict a spatial affine correction with ONNX Resize/Sigmoid/Clip only.

    ``2*sigmoid(2*z)-1`` replaces tanh because Tanh is outside the organizer's
    operator vocabulary. The zero initialized projection makes y0 exactly x.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(ConvBNAct(3, 16, 3, 2),
                                      ConvBNAct(16, 32, 3, 2),
                                      ConvBNAct(32, 64, 3, 2))
        self.affine = nn.Conv2d(64, 6, 1)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x):
        thumbnail = F.interpolate(x, size=(64, 128), mode="bilinear", align_corners=False)
        maps = self.affine(self.features(thumbnail))
        # Export is deliberately static; Python ints prevent a Shape-driven
        # Resize graph from hiding downstream convolution FLOPs in the checker.
        maps = F.interpolate(maps, size=(int(x.shape[-2]), int(x.shape[-1])),
                             mode="bilinear", align_corners=False)
        gain = torch.clamp(1.0 + 1.5 * (2.0 * torch.sigmoid(2.0 * maps[:, :3]) - 1.0), min=0.1)
        offset = 0.5 * (2.0 * torch.sigmoid(2.0 * maps[:, 3:]) - 1.0)
        corrected = torch.clamp(gain * x + offset, 0.0, 1.0)
        return corrected.reshape(-1, 3, int(x.shape[-2]), int(x.shape[-1]))


class _DuoLiteHeads(nn.Module):
    """DuoLite-S1의 디코더·context 헤드·복원 분기를 인코더 채널에 맞춰 구성한다.

    forward(x, s0, s1, s2, s3, s4)로 인코더 출력을 받아 DuoLite-S1과 동일한 dict를 낸다.
    s0: H/2(pre_stem), s1~s4: stride 4/8/16/32(timm 백본).
    """

    def __init__(self, enc_ch: List[int], stem_ch: int, dec: int, cfg: StudentConfig,
                 decoder: str = "fpn_lite", seg_head: str = "flat",
                 restore_grad_block_deep: bool = False,
                 restore_head: str = "global", cell_stride: int = 32,
                 seg_couple: str = "none", couple_control: str = "none",
                 couple_detach: bool = False):
        super().__init__()
        self.cfg = cfg
        c1, c2, c3, c4 = enc_ch
        self.dec = dec
        self.decoder = decoder
        self.seg_head_type = seg_head
        self.restore_grad_block_deep = restore_grad_block_deep
        self.restore_head = restore_head
        self.cell_stride = cell_stride
        # 복원->분할 결합(docs/seg_couple.md). none이면 기존 출력과 동일하다.
        self.seg_couple = seg_couple
        self.couple_control = couple_control
        self.couple_detach = couple_detach
        self.couple_film_on = "film" in seg_couple
        self.couple_img_on = "img" in seg_couple
        if (self.couple_film_on or self.couple_img_on) and decoder != "enet_dec":
            raise ValueError(f"seg_couple={seg_couple!r} is only implemented for enet_dec, "
                             f"got decoder {decoder!r}")

        # ---- H/2 복원용 자체 stem(백본이 stride 2를 주지 않음) ---------- #
        self.pre_stem = ConvBNAct(3, stem_ch, 3, 2)

        # ---- global context + image-level 헤드 -------------------------- #
        self.gctx = GlobalContext(c4)
        self.ctx_fc = nn.Sequential(nn.Conv2d(c4, cfg.ctx_hidden, 1), nn.ReLU(inplace=True))
        self.presence_head = nn.Conv2d(cfg.ctx_hidden, cfg.n_classes, 1)
        self.medium_head = nn.Conv2d(cfg.ctx_hidden, cfg.n_water + 1, 1)
        n_glob = 3 + 3 + 9 + 3 * cfg.tone_knots
        self.restore_global = nn.Conv2d(cfg.ctx_hidden, n_glob, 1)
        nn.init.zeros_(self.restore_global.weight); nn.init.zeros_(self.restore_global.bias)
        if cfg.film:
            self.film = nn.Conv2d(cfg.ctx_hidden, 2 * dec, 1)
            nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        # 복원->분할 결합 모듈(공동 학습; glob에 stop-grad 없음, couple_detach로만 건다).
        # film: 전역 복원 파라미터 glob(n_glob,1,1)을 1x1 conv로 2*dec γ/β로 바꿔(0 초기화,
        #       초기 항등) stride16(p3)·stride8(p2) 디코더 출력에 공유 적용한다.
        if self.couple_film_on:
            self.couple_film = nn.Conv2d(n_glob, 2 * dec, 1)
            nn.init.zeros_(self.couple_film.weight); nn.init.zeros_(self.couple_film.bias)
        # img: 1/4 보정 이미지의 3채널을 3x3 conv로 dec/4채널로 만든 뒤
        #      p1 bottleneck 입력에 concat한다. 첫 bottleneck만 확장 입력을 받는다.
        if self.couple_img_on:
            img_mid = max(1, dec // 4)
            self.couple_img = nn.Conv2d(3, img_mid, 3, padding=1, bias=False)
            nn.init.normal_(self.couple_img.weight, std=1e-2)
            # 1/4 평균 풀을 고정 grouped Conv로 둔다: ONNX AveragePool(어휘 밖) 대신
            # Conv(어휘 내)로 export되고, 학습 파라미터가 아니며 정적 shape다.
            pool_w = torch.full((3, 1, 4, 4), 1.0 / 16.0)
            self.register_buffer("couple_pool_w", pool_w)

        # ---- 분할 디코더: context/복원/FiLM 경로는 공통 ------------------ #
        if decoder in ("fpn_lite", "fpn_wide"):
            self.lat4 = ConvBNAct(c4, dec, 1, act=False)
            self.lat3 = ConvBNAct(c3, dec, 1, act=False)
            self.lat2 = ConvBNAct(c2, dec, 1, act=False)
            self.lat1 = ConvBNAct(c1, dec, 1, act=False)
            self.ref3 = DWSep(dec)
            self.ref2 = DWSep(dec)
            if decoder == "fpn_wide":
                # Deep dense refinement spends capacity at low spatial cost.
                self.deep_refine = nn.Sequential(ConvBNAct(dec, dec, 3),
                                                 ConvBNAct(dec, dec, 3),
                                                 DWSep(dec), DWSep(dec), DWSep(dec))
                self.ref1 = nn.Sequential(DWSep(dec), DWSep(dec))
            else:
                self.ref1 = ConvBNAct(dec, dec, 3) if cfg.decoder_full_conv else DWSep(dec)
        elif decoder == "fpe":
            self.fpe4 = _FPEEncoding(c4, dec)
            self.fpe3 = _FPEEncoding(c3, dec)
            self.fpe2 = _FPEEncoding(c2, dec)
            self.fpe1 = _FPEEncoding(c1, dec)
            self.mea3 = _MEAFusion(dec)
            self.mea2 = _MEAFusion(dec)
            self.mea1 = _MEAFusion(dec)
            self.fpe_deep_refine = ConvBNAct(dec, dec, 3)
            self.fpe_refine = nn.Sequential(DWSep(dec), DWSep(dec))
        elif decoder == "enet_dec":
            self.enet_lat4 = ConvBNAct(c4, dec, 1, act=False)
            self.enet_skips = nn.ModuleList(ConvBNAct(ch, dec, 1, act=False)
                                            for ch in (c3, c2, c1))
            self.enet_stages = nn.ModuleList(
                nn.Sequential(_ENetBottleneck(dec, img_mid if self.couple_img_on and i == 3 else 0),
                              _ENetBottleneck(dec)) for i in range(4))
        else:
            raise ValueError(f"Unknown decoder {decoder!r}")
        if seg_head == "hierarchical":
            self.region_head = nn.Conv2d(dec, 2, 1)  # animal and coral|nonanimal logits
            self.kind_head = nn.Conv2d(dec, 3, 1)    # fish, elasmo, other scores
        else:
            self.seg_head = nn.Conv2d(dec, cfg.n_classes, 1)
        self.class_bias = nn.Parameter(torch.zeros(cfg.n_classes))
        self.presence_gain = nn.Parameter(torch.zeros(1, cfg.n_classes, 1, 1))

        # ---- 복원 분기 -------------------------------------------------- #
        r = cfg.res_local
        self.res_lat = ConvBNAct(c1, r, 1)
        self.res_mix = nn.Sequential(ConvBNAct(r + stem_ch, r + stem_ch, 3, groups=r + stem_ch),
                                     ConvBNAct(r + stem_ch, r, 1))
        self.res_out = nn.Conv2d(r, 3, 1)
        nn.init.zeros_(self.res_out.weight); nn.init.zeros_(self.res_out.bias)
        self.refiner = nn.Sequential(nn.Conv2d(6, cfg.refiner, 3, 1, 1), nn.ReLU(inplace=True),
                                     nn.Conv2d(cfg.refiner, 3, 3, 1, 1))
        nn.init.zeros_(self.refiner[2].weight); nn.init.zeros_(self.refiner[2].bias)
        if restore_head == "cell_affine":
            self.cell_affine = nn.Conv2d(dec, 6, 1)
            nn.init.zeros_(self.cell_affine.weight)
            nn.init.zeros_(self.cell_affine.bias)

    def _global_correct(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """DuoLite-S1._global_correct와 동일한 구조화 전역 색 보정(예측 0이면 항등)."""
        K = self.cfg.tone_knots
        gain = 1.0 + p[:, 0:3]
        bias = p[:, 3:6]
        ccm = p[:, 6:15]
        slopes = 2.0 * torch.sigmoid(p[:, 15:15 + 3 * K]) - 1.0
        y = x * gain + bias
        r, g, b = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        r2 = r * (1.0 + ccm[:, 0:1]) + g * ccm[:, 1:2] + b * ccm[:, 2:3]
        g2 = r * ccm[:, 3:4] + g * (1.0 + ccm[:, 4:5]) + b * ccm[:, 5:6]
        b2 = r * ccm[:, 6:7] + g * ccm[:, 7:8] + b * (1.0 + ccm[:, 8:9])
        y = torch.cat([r2, g2, b2], dim=1)
        y = torch.clamp(y, 0.0, 1.0)
        out = y
        seg_w = 1.0 / K
        for k in range(K):
            s_k = slopes[:, 3 * k:3 * k + 3]
            out = out + s_k * torch.clamp(y - k * seg_w, 0.0, seg_w)
        return out

    def _cell_correct(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Apply interpolated per-cell RGB gain and offset at a static export size."""
        if self.restore_grad_block_deep:
            features = features.detach()
        maps = self.cell_affine(features)
        maps = F.interpolate(maps, size=(int(x.shape[-2]), int(x.shape[-1])),
                             mode="bilinear", align_corners=False)
        gain = 1.0 + 1.5 * (2.0 * torch.sigmoid(2.0 * maps[:, :3]) - 1.0)
        offset = 0.5 * (2.0 * torch.sigmoid(2.0 * maps[:, 3:]) - 1.0)
        return gain * x + offset

    def _shuffle_derangement(self, b: int, device) -> torch.Tensor:
        """배치>=2에서 자기 자신을 고르지 않는(고정점 없는) 순열을 뽑는다(거부 샘플링).

        glob/base에 **같은** 순열을 적용하려고 forward에서 한 번만 호출해 공유한다.
        shuffle은 학습·검증(eager)에서만 쓰고 export는 건너뛰므로 Python 루프가 안전하다.
        """
        idx = torch.arange(b, device=device)
        while True:
            perm = torch.randperm(b, device=device)
            if not bool((perm == idx).any()):
                return perm

    def _couple_glob(self, glob: torch.Tensor, perm) -> torch.Tensor:
        """film 결합에 넣을 전역 파라미터를 control/detach에 맞춰 고른다."""
        v = glob.detach() if self.couple_detach else glob
        if self.couple_control == "identity":
            return torch.zeros_like(v)       # 복원 정보 제거(구조·FLOPs 동일, K1)
        if perm is not None:
            return v[perm]
        return v

    def _couple_img_pooled(self, x: torch.Tensor, glob_base: torch.Tensor,
                           perm) -> torch.Tensor:
        """img 결합에 넣을 보정 이미지를 control/detach에 맞춰 1/4로 avg-pool한다.

        shuffle은 glob과 **같은** ``perm``을 쓴다.
        """
        if self.couple_control == "identity":
            src = x                           # base 대신 열화 입력(복원 정보 제거, K1)
        else:
            src = glob_base.detach() if self.couple_detach else glob_base
            if perm is not None:
                src = src[perm]
        # 고정 grouped Conv = 정확한 4x4 평균 풀(동적 Resize 없음, 어휘 내 Conv op).
        return F.conv2d(src, self.couple_pool_w, stride=4, groups=3)

    def forward(self, x, s0, s1, s2, s3, s4) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        # Restoration uses s1 locally and s4 for global color. s3 is never used
        # by restoration. A detached s4 keeps restoration gradients out of the
        # deep backbone while segmentation still sees the original s4 graph.
        if self.restore_grad_block_deep:
            restore_g = gap_matmul(self.gctx(s4.detach(), gap_matmul(s4.detach())))
        g = gap_matmul(s4)
        s4 = self.gctx(s4, g)
        g = gap_matmul(s4)

        h = self.ctx_fc(g)
        presence = self.presence_head(h)
        medium = self.medium_head(h)
        if self.restore_grad_block_deep:
            restore_h = self.ctx_fc(restore_g)
            glob = self.restore_global(restore_h)
        else:
            glob = self.restore_global(h)

        # 전역 보정 이미지: img 결합과 global 복원이 공유한다(이중 계산 방지).
        glob_base = None
        if self.couple_img_on or self.restore_head != "cell_affine":
            glob_base = self._global_correct(x, glob)

        d = self.dec
        if self.decoder in ("fpn_lite", "fpn_wide"):
            p4 = self.lat4(s4)
            if self.decoder == "fpn_wide":
                p4 = self.deep_refine(p4)
            p3 = self.ref3(self.lat3(s3) + up2(p4))
            p2 = self.lat2(s2) + up2(p3)
            if cfg.film:
                fb = self.film(h)
                p2 = p2 * (1.0 + fb[:, :d]) + fb[:, d:]
            p2 = self.ref2(p2)
            p1 = self.ref1(self.lat1(s1) + up2(p2))
        elif self.decoder == "fpe":
            p4 = self.fpe_deep_refine(self.fpe4(s4))
            p3 = self.mea3(p4, self.fpe3(s3))
            p2 = self.mea2(p3, self.fpe2(s2))
            if cfg.film:
                fb = self.film(h)
                p2 = p2 * (1.0 + fb[:, :d]) + fb[:, d:]
            p1 = self.fpe_refine(self.mea1(p2, self.fpe1(s1)))
        else:
            # K2는 매 forward마다 배치 안의 다른 이미지와 무작위로 짝짓는다.
            shuffle_perm = None
            if self.couple_control == "shuffle" and (self.couple_film_on or self.couple_img_on):
                b = x.shape[0]
                if b < 2:
                    raise ValueError("couple_control='shuffle' requires at least two distinct images per forward")
                shuffle_perm = self._shuffle_derangement(b, x.device)
            # 복원->분할 film 결합용 공유 γ/β(glob 1x1 conv, 0 초기화 -> 초기 항등).
            couple_gamma = couple_beta = None
            if self.couple_film_on:
                cfb = self.couple_film(self._couple_glob(glob, shuffle_perm))
                couple_gamma, couple_beta = cfb[:, :d], cfb[:, d:]
            p4 = self.enet_stages[0](self.enet_lat4(s4))
            p3 = self.enet_stages[1](self.enet_skips[0](s3) + F.interpolate(
                p4, size=s3.shape[-2:], mode="bilinear", align_corners=False))
            if self.couple_film_on:                       # stride16 출력 변조
                p3 = p3 * (1.0 + couple_gamma) + couple_beta
            p2 = self.enet_skips[1](s2) + F.interpolate(
                p3, size=s2.shape[-2:], mode="bilinear", align_corners=False)
            if cfg.film:
                fb = self.film(h)
                p2 = p2 * (1.0 + fb[:, :d]) + fb[:, d:]
            p2 = self.enet_stages[2](p2)
            if self.couple_film_on:                       # stride8 출력 변조(같은 γ/β)
                p2 = p2 * (1.0 + couple_gamma) + couple_beta
            s1_feat = self.enet_skips[2](s1) + F.interpolate(
                p2, size=s1.shape[-2:], mode="bilinear", align_corners=False)
            if self.couple_img_on:                        # stride4 bottleneck 입력 concat
                s1_feat = torch.cat((s1_feat, self.couple_img(
                    self._couple_img_pooled(x, glob_base, shuffle_perm))), dim=1)
            p1 = self.enet_stages[3](s1_feat)
        extra = {}
        if self.seg_head_type == "hierarchical":
            # No Softmax/Exp/Log: sigmoid gates and normalized one-vs-rest
            # sigmoid kinds use only organizer operators. Calibration acts as
            # a positive probability multiplier, then is renormalized.
            region = F.interpolate(self.region_head(p1), scale_factor=4.0,
                                   mode="bilinear", align_corners=False)
            kind = F.interpolate(self.kind_head(p1), scale_factor=4.0,
                                 mode="bilinear", align_corners=False)
            animal = torch.sigmoid(region[:, :1])
            coral_given_nonanimal = torch.sigmoid(region[:, 1:2])
            kinds = torch.sigmoid(kind)
            kinds = kinds / (kinds[:, :1] + kinds[:, 1:2] + kinds[:, 2:3])
            probs = torch.cat(((1.0 - animal) * (1.0 - coral_given_nonanimal),
                               (1.0 - animal) * coral_given_nonanimal,
                               animal * kinds), dim=1)
            multiplier = torch.clamp(1.0 + self.class_bias.reshape(1, 5, 1, 1)
                                     + self.presence_gain * torch.sigmoid(presence), min=0.01)
            probs = probs * multiplier
            seg_logits = probs / (probs[:, :1] + probs[:, 1:2] + probs[:, 2:3]
                                  + probs[:, 3:4] + probs[:, 4:5])
            extra = {"region_logits": region, "kind_logits": kind,
                     "seg_probabilities": True}
        else:
            logits = self.seg_head(p1)
            logits = logits + self.class_bias.reshape(1, cfg.n_classes, 1, 1) \
                + self.presence_gain * torch.sigmoid(presence)
            seg_logits = F.interpolate(logits, scale_factor=4.0, mode="bilinear", align_corners=False)

        if self.restore_head == "cell_affine":
            base = self._cell_correct(x, p4 if self.cell_stride == 32 else p3)
        else:
            base = glob_base  # 위에서 한 번만 계산해 재사용
        loc = self.res_mix(torch.cat([up2(self.res_lat(s1)), s0], dim=1))
        resid = F.interpolate(self.res_out(loc), scale_factor=2.0, mode="bilinear", align_corners=False)
        y = base + resid
        y = y + self.refiner(torch.cat([x, y], dim=1))
        restored = torch.clamp(y, 0.0, 1.0)

        return {"restored_image": restored, "seg_logits": seg_logits, **extra,
                "presence_logits": presence.flatten(1),
                "water_logits": medium[:, :cfg.n_water].flatten(1),
                "depth": medium[:, cfg.n_water:].flatten(1)}


class DuoLitePre(nn.Module):
    """timm 사전학습 인코더 + DuoLite-S1 디코더/헤드 변형.

    forward(x) -> dict(restored_image, seg_logits, presence_logits(B,5),
                       water_logits(B,10), depth(B,1))
    forward_export(x) -> (restored_image, seg_logits)

    ``mobilenetv2_050`` + ``enet_dec`` + ``cell_affine`` (stride 32),
    512x1024 static simplified ONNX: 979,606 PyTorch params, 0.9557M ONNX
    params, 11.6726 GFLOPs. At stride 16: same params, 11.6765 GFLOPs.
    Both pass the strict organizer gate (see ``docs/cell_affine.md``).

    ``seg_couple`` (enet_dec 전용, 기본 none)는 복원 분기를 분할 디코더에 공동 학습으로
    결합한다(docs/seg_couple.md). ``film``은 전역 복원 파라미터 glob을 1x1 conv(0 초기화)
    로 γ/β로 바꿔 stride16/8 디코더 출력을 변조하고, ``img``는 전역 보정 이미지 base를
    1/4(고정 grouped Conv 평균 풀, AveragePool 아님)로 줄여 3x3 conv(3->dec/4,
    소폭 초기화)를 거쳐 stride4 p1 bottleneck의 입력에 concat한다. 첫 bottleneck은
    확장된 입력을 받고 dec채널 잔차를 보존한다. ``film+img``는 둘 다.
    ``couple_control`` identity(K1)·shuffle(K2)는
    구조·FLOPs를 유지하며 복원 정보만 제거/교란하는 대조군이고, ``couple_detach``는
    glob/base에 stop-grad를 건다. none은 기존 enet_dec 출력과 byte 단위로 동일하다.
    shuffle(K2)은 glob/base에 같은 무작위 derangement를 쓴다(배치>=2 필수).
    검증 루프는 매 샘플에 다른 무작위 이미지를 붙여 배치 2장으로 평가한다.
    512x1024 strict simplified ONNX 실측(mobilenetv2_050, global restore,
    flat seg, pretrained=False; PyTorch params / ONNX params / GFLOPs):
    none 978,328 / 0.9569M / 11.6713; film 979,060 / 0.9574M / 11.3526;
    img 982,568 / 0.9612M / 11.9523; film+img 983,168 / 0.9615M / 11.6249.
    네 모드 모두 1.0M·12 GFLOPs 이내, strict organizer gate PASSED.
    """

    def __init__(self, backbone: str = "tf_mobilenetv3_small_minimal_100",
                 pretrained: bool = True, width: float = 1.0, dec: Optional[int] = None,
                 decoder: str = "fpn_lite", correct_then_segment: bool = False,
                 ctseg_stopgrad: bool = True, seg_head: str = "flat",
                 restore_grad_block_deep: bool = False,
                 restore_head: str = "global", cell_stride: int = 32,
                 seg_couple: str = "none", couple_control: str = "none",
                 couple_detach: bool = False, export_safe: bool = False):
        super().__init__()
        import timm  # 지연 import: timm 없는 환경에서 다른 모델 사용을 막지 않는다.

        self.backbone_name = backbone
        if decoder not in ("fpn_lite", "fpn_wide", "fpe", "enet_dec"):
            raise ValueError(f"Unknown decoder {decoder!r}")
        self.decoder = decoder
        if restore_head not in ("global", "cell_affine"):
            raise ValueError(f"Unknown restoration head {restore_head!r}")
        if cell_stride not in (16, 32):
            raise ValueError(f"cell_stride must be 16 or 32, got {cell_stride!r}")
        if seg_couple not in ("none", "film", "film+img", "img"):
            raise ValueError(f"Unknown seg_couple {seg_couple!r}")
        if couple_control not in ("none", "identity", "shuffle"):
            raise ValueError(f"Unknown couple_control {couple_control!r}")
        self.restore_head = restore_head
        self.cell_stride = cell_stride
        self.seg_couple = seg_couple
        self.couple_control = couple_control
        self.couple_detach = bool(couple_detach)
        self.pretrained = bool(pretrained)
        if seg_head not in ("flat", "hierarchical"):
            raise ValueError(f"Unknown segmentation head {seg_head!r}")
        self.correct_then_segment = bool(correct_then_segment)
        self.ctseg_stopgrad = bool(ctseg_stopgrad)
        self.restore_grad_block_deep = bool(restore_grad_block_deep)
        self.corrector = _AffineCorrector() if self.correct_then_segment else None
        self.cfg = cfg = _scaled_config(float(width))
        self.backbone = timm.create_model(backbone, pretrained=bool(pretrained),
                                          features_only=True, out_indices=(1, 2, 3, 4))
        # export_safe=True(SM8): LCNet·MobileNetV3 계열의 HardSwish·HardSigmoid·SE 평균을 값이 같은
        # 허용 op 식으로 바꾼다. 기본 False = 기존 동작(어휘 밖 op가 남아 strict 게이트가 거부).
        self.export_safe = bool(export_safe)
        if self.export_safe:
            from .student_v2 import make_export_safe
            make_export_safe(self.backbone)
        enc_ch = list(self.backbone.feature_info.channels())
        if len(enc_ch) != 4:
            raise ValueError(f"{backbone!r} features_only out_indices=(1,2,3,4) must yield 4 "
                             f"feature maps, got channels {enc_ch}")
        self._enc_ch = enc_ch
        stem_ch = cfg.stem

        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        variant_widths = {"fpn_wide": 128, "fpe": 160, "enet_dec": 212}
        if dec is not None:
            candidates = [int(dec)]
        elif decoder == "fpn_lite":
            candidates = list(_DEC_CANDIDATES)
        else:
            candidates = [variant_widths[decoder]]
            if (self.corrector is not None or seg_head == "hierarchical"
                    or restore_head == "cell_affine" or seg_couple != "none"):
                candidates += [208, 204, 200, 196, 192, 184, 176, 160]
        chosen = None
        for dec_try in candidates:
            heads = _DuoLiteHeads(enc_ch, stem_ch, dec_try, cfg, decoder, seg_head,
                                  self.restore_grad_block_deep, restore_head, cell_stride,
                                  seg_couple, couple_control, self.couple_detach)
            total = backbone_params + sum(p.numel() for p in heads.parameters()) + (sum(
                p.numel() for p in self.corrector.parameters()) if self.corrector else 0)
            if dec is not None or total <= _PARAM_CAP * _PARAM_SAFETY:
                chosen, self.heads = dec_try, heads
                break
            chosen, self.heads = dec_try, heads  # 마지막 후보라도 보관(캡 초과면 아래서 알림)
        self.dec_used = chosen
        total = sum(p.numel() for p in self.parameters())
        if total > _PARAM_CAP:
            raise ValueError(
                f"DuoLitePre({backbone!r}, decoder={decoder!r}) has {total:,} params "
                f"(>1.0M cap) at dec={chosen}. 더 작은 백본을 쓰거나 dec를 "
                f"직접 낮춰라(현재 백본 {backbone_params:,} params).")

    # 캘리브레이션 유틸이 참조하는 class_bias를 최상위에서 노출한다.
    @property
    def class_bias(self) -> nn.Parameter:
        return self.heads.class_bias

    def encoder_parameters(self):
        """사전학습 인코더 파라미터(train.py가 낮은 lr 그룹으로 묶는다)."""
        return self.backbone.parameters()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        y0 = self.corrector(x) if self.corrector is not None else x
        encoder_input = y0.detach() if self.corrector is not None and self.ctseg_stopgrad else y0
        s1, s2, s3, s4 = self.backbone(encoder_input)
        s0 = self.heads.pre_stem(y0)
        output = self.heads(y0, s0, s1, s2, s3, s4)
        if self.training:
            output["enc_feats"] = (s2, s3, s4)
        if self.corrector is not None:
            output["y0"] = y0
        return output

    def forward_export(self, x: torch.Tensor):
        # export는 control(identity/shuffle)을 무시하고 실제 결합 경로를 쓴다.
        saved = self.heads.couple_control
        self.heads.couple_control = "none"
        try:
            outputs = self.forward(x)
        finally:
            self.heads.couple_control = saved
        return outputs["restored_image"], outputs["seg_logits"]
