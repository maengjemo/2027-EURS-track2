"""Track 2 학생 v2: 설계 1(DS-FPN)·2(Detail/Context 2분기)·3(병목 어텐션 + 반복 복원).

세 모델 모두 DuoLitePre와 같은 인터페이스를 따른다.
forward(x) -> dict(restored_image, seg_logits, presence_logits(B,5), water_logits(B,10), depth(B,1))
forward_export(x) -> (restored_image, seg_logits)

주최측 허용 op만으로 export되도록 원 설계에서 바꾼 부분(docs/student_v2.md):
- PPM의 적응 평균 풀링: AveragePool은 어휘 밖이고 1/32 특징이 bin으로 나누어떨어지지 않을 수
  있다(640x768 -> 20x24). AdaptiveAvgPool2d와 같은 구간의 행·열 분리 상수 MatMul로 계산한다.
- PixelShuffle: ONNX DepthToSpace(어휘 밖) 대신 Reshape·Transpose로 같은 재배열을 한다.
- 설계 3 MHSA: Softmax·LayerNorm·GELU가 어휘 밖이라 기본값은 ReLU 선형 어텐션(EfficientViT식,
  MatMul·Div·ReLU), BatchNorm, ReLU MLP다. attn="softmax"는 비교 실험용이며 strict export에서 거부된다.
- RepConv: 학습 때 3x3+1x1(+identity) BN 분기 합, export 직전 switch_to_deploy()로 단일 3x3.
"""
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .duolite_s1 import ConvBNAct, GlobalContext, _scaled_config, gap_matmul


def _resize(x: torch.Tensor, ref) -> torch.Tensor:
    """정적 크기 bilinear Resize. ref는 텐서(뒤 두 축 크기를 따름) 또는 (h, w)."""
    size = ref.shape[-2:] if isinstance(ref, torch.Tensor) else ref
    return F.interpolate(x, size=(int(size[0]), int(size[1])), mode="bilinear", align_corners=False)


def _film(x: torch.Tensor, gb: torch.Tensor) -> torch.Tensor:
    c = x.shape[1]
    return x * (1.0 + gb[:, :c]) + gb[:, c:]


def _zero_conv(cin: int, cout: int) -> nn.Conv2d:
    conv = nn.Conv2d(cin, cout, 1)
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv


def global_correct(x: torch.Tensor, p: torch.Tensor, knots: int) -> torch.Tensor:
    """DuoLite 전역 색 보정(_DuoLiteHeads._global_correct와 같은 식). p=0이면 항등."""
    gain = 1.0 + p[:, 0:3]
    bias = p[:, 3:6]
    ccm = p[:, 6:15]
    slopes = 2.0 * torch.sigmoid(p[:, 15:15 + 3 * knots]) - 1.0
    y = x * gain + bias
    r, g, b = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    r2 = r * (1.0 + ccm[:, 0:1]) + g * ccm[:, 1:2] + b * ccm[:, 2:3]
    g2 = r * ccm[:, 3:4] + g * (1.0 + ccm[:, 4:5]) + b * ccm[:, 5:6]
    b2 = r * ccm[:, 6:7] + g * ccm[:, 7:8] + b * (1.0 + ccm[:, 8:9])
    y = torch.clamp(torch.cat([r2, g2, b2], dim=1), 0.0, 1.0)
    out = y
    seg_w = 1.0 / knots
    for k in range(knots):
        out = out + slopes[:, 3 * k:3 * k + 3] * torch.clamp(y - k * seg_w, 0.0, seg_w)
    return out


def _pool_matrix(n_in: int, bins: int, device, dtype) -> torch.Tensor:
    """AdaptiveAvgPool 1축 구간(start=floor(i*n/b), end=ceil((i+1)*n/b))의 평균 행렬 (n_in, bins).

    파이썬 리스트로 만들어 export 그래프에 상수 하나로 남긴다(텐서 부분 대입은 ScatterND가 된다).
    """
    rows = [[0.0] * bins for _ in range(n_in)]
    for i in range(bins):
        start, end = (i * n_in) // bins, -((-(i + 1) * n_in) // bins)
        for r in range(start, end):
            rows[r][i] = 1.0 / (end - start)
    return torch.tensor(rows, device=device, dtype=dtype)


def adaptive_avg_pool_mm(x: torch.Tensor, bins: int) -> torch.Tensor:
    """F.adaptive_avg_pool2d(x, bins)와 같은 값을 행·열 분리 상수 MatMul로 계산한다.

    ONNX에는 MatMul·Transpose만 남고, 상수는 (H+W)·bins개라 파라미터 집계에 거의 영향이 없다.
    """
    h, w = int(x.shape[-2]), int(x.shape[-1])
    y = torch.matmul(x, _pool_matrix(w, bins, x.device, x.dtype))                    # (n, c, h, b)
    y = torch.matmul(y.transpose(2, 3), _pool_matrix(h, bins, x.device, x.dtype))    # (n, c, b_w, b_h)
    return y.transpose(2, 3)


def pixel_shuffle2(x: torch.Tensor) -> torch.Tensor:
    """F.pixel_shuffle(x, 2)와 같은 재배열을 Reshape·Transpose로 한다(DepthToSpace 회피)."""
    n, c4, h, w = (int(v) for v in x.shape)
    c = c4 // 4
    x = x.reshape(n, c, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
    return x.reshape(n, c, h * 2, w * 2)


class DSBlock(nn.Module):
    """잔차 depthwise-separable 블록: ReLU(x + BN(PW(ReLU(BN(DW3x3(x))))))."""

    def __init__(self, c: int):
        super().__init__()
        self.dw = ConvBNAct(c, c, 3, groups=c)
        self.pw = ConvBNAct(c, c, 1, act=False)

    def forward(self, x):
        return F.relu(x + self.pw(self.dw(x)))


def _fuse_bn(kernel: torch.Tensor, bn: nn.BatchNorm2d):
    std = torch.sqrt(bn.running_var + bn.eps)
    t = bn.weight / std
    return kernel * t.reshape(-1, 1, 1, 1), bn.bias - bn.running_mean * t


class RepConv(nn.Module):
    """RepVGG식 3x3 conv. 학습: 3x3+BN, 1x1+BN, (cin==cout·stride 1이면) identity BN의 합.

    switch_to_deploy()가 세 분기를 단일 3x3 conv(bias 포함)로 합친다. eval 모드 출력은 합치기
    전후가 같다. zero_init=True면 모든 분기 BN의 weight를 0으로 두어 처음 출력이 0이다.
    """

    def __init__(self, cin: int, cout: int, stride: int = 1, act: bool = True, zero_init: bool = False):
        super().__init__()
        self.cin, self.cout, self.stride, self.act = cin, cout, stride, act
        self.k3 = nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout))
        self.k1 = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, 0, bias=False), nn.BatchNorm2d(cout))
        self.idb = nn.BatchNorm2d(cout) if cin == cout and stride == 1 else None
        self.fused = None
        if zero_init:
            for bn in (self.k3[1], self.k1[1], self.idb):
                if bn is not None:
                    nn.init.zeros_(bn.weight)

    def forward(self, x):
        if self.fused is not None:
            y = self.fused(x)
        else:
            y = self.k3(x) + self.k1(x)
            if self.idb is not None:
                y = y + self.idb(x)
        return F.relu(y) if self.act else y

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.fused is not None:
            return
        w, b = _fuse_bn(self.k3[0].weight, self.k3[1])
        w1, b1 = _fuse_bn(self.k1[0].weight, self.k1[1])
        w, b = w + F.pad(w1, [1, 1, 1, 1]), b + b1
        if self.idb is not None:
            ident = torch.zeros_like(w)
            idx = torch.arange(self.cout, device=w.device)
            ident[idx, idx, 1, 1] = 1.0
            wi, bi = _fuse_bn(ident, self.idb)
            w, b = w + wi, b + bi
        conv = nn.Conv2d(self.cin, self.cout, 3, self.stride, 1, bias=True).to(w.device)
        conv.weight.copy_(w)
        conv.bias.copy_(b)
        self.fused = conv
        self.k3 = self.k1 = self.idb = None


class PPM(nn.Module):
    """Pyramid pooling(bin 1/2/3/6, 각 1x1->branch) + 입력과 concat -> 1x1 fuse.

    bin 가지는 BN 없이 Conv+ReLU다(1x1 풀링 출력은 batch 1 학습에서 BN 통계를 낼 수 없다).
    """

    def __init__(self, cin: int, branch: int = 32, bins: Sequence[int] = (1, 2, 3, 6), cout: int = 64):
        super().__init__()
        self.bins = tuple(int(b) for b in bins)
        self.stages = nn.ModuleList(nn.Sequential(nn.Conv2d(cin, branch, 1), nn.ReLU(inplace=True))
                                    for _ in self.bins)
        self.fuse = ConvBNAct(cin + branch * len(self.bins), cout, 1)

    def forward(self, x):
        feats = [x] + [_resize(stage(adaptive_avg_pool_mm(x, b)), x)
                       for b, stage in zip(self.bins, self.stages)]
        return self.fuse(torch.cat(feats, dim=1))


class LiteAttention(nn.Module):
    """병목 self-attention 블록(설계 3): CPE(DW3x3) -> attention -> MLP, 둘 다 잔차.

    attn="relu_linear"(기본): q·k에 ReLU를 걸고 out = q(kᵀv) / q(kᵀ1) (EfficientViT LiteMLA식).
    ONNX에는 MatMul·Div·ReLU·Concat·Slice만 남는다. attn="softmax"는 표준 scaled dot-product
    (ONNX Softmax가 생겨 어휘 밖). 정규화는 BatchNorm, MLP 활성은 ReLU다. proj와 MLP 출력은
    0 초기화라 사전학습 인코더 뒤에 넣어도 처음에는 CPE만 더한 항등에 가깝다.
    """

    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 2.0, attn: str = "relu_linear"):
        super().__init__()
        if attn not in ("relu_linear", "softmax"):
            raise ValueError(f"Unknown attn {attn!r}")
        if dim % heads:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.heads, self.attn = heads, attn
        self.cpe = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.qkv = nn.Conv2d(dim, 3 * dim, 1)
        self.proj = _zero_conv(dim, dim)
        self.norm2 = nn.BatchNorm2d(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.ReLU(inplace=True), _zero_conv(hidden, dim))

    def forward(self, x):
        x = x + self.cpe(x)
        n, c, h, w = (int(v) for v in x.shape)
        d = c // self.heads
        qkv = self.qkv(self.norm1(x)).reshape(n, 3, self.heads, d, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]                 # (n, heads, d, N)
        if self.attn == "relu_linear":
            q, k = F.relu(q), F.relu(k)
            v1 = torch.cat([v, v[:, :, :1] * 0.0 + 1.0], dim=2)     # (n, heads, d+1, N), 마지막 행 = 1
            out = torch.matmul(torch.matmul(v1, k.transpose(-1, -2)), q)  # (n, heads, d+1, N)
            out = out[:, :, :d] / (out[:, :, d:] + 1e-6)
        else:
            scores = torch.matmul(q.transpose(-1, -2), k) * (d ** -0.5)  # (n, heads, N, N)
            out = torch.matmul(v, torch.softmax(scores, dim=-1).transpose(-1, -2))
        x = x + self.proj(out.reshape(n, c, h, w))
        return x + self.mlp(self.norm2(x))


class _HardSwish(nn.Module):
    """nn.Hardswish와 같은 값: x·clip(x+3, 0, 6)·(1/6) (ONNX: Add·Clip·Mul — Div 없이 S10이 컴파일한 op 집합 안)."""

    def forward(self, x):
        return x * (torch.clamp(x + 3.0, 0.0, 6.0) * (1.0 / 6.0))


class _HardSigmoid(nn.Module):
    """nn.Hardsigmoid와 같은 값: clip(x+3, 0, 6)·(1/6)."""

    def forward(self, x):
        return torch.clamp(x + 3.0, 0.0, 6.0) * (1.0 / 6.0)


class _SafeSqueezeExcite(nn.Module):
    """timm SqueezeExcite와 같은 계산에서 전역 평균(ReduceMean)만 gap_matmul로 바꾼다.

    하위 모듈 이름(conv_reduce·act1·conv_expand·gate)을 그대로 두어 state_dict 키가 같다.
    """

    def __init__(self, se: nn.Module):
        super().__init__()
        self.conv_reduce, self.act1 = se.conv_reduce, se.act1
        self.conv_expand, self.gate = se.conv_expand, se.gate

    def forward(self, x):
        return x * self.gate(self.conv_expand(self.act1(self.conv_reduce(gap_matmul(x)))))


def make_export_safe(module: nn.Module) -> nn.Module:
    """timm 인코더의 어휘 밖 op(HardSwish·HardSigmoid·SE의 ReduceMean)를 값이 같은 허용 op 식으로 바꾼다.

    사전학습 가중치는 그대로라 LCNet·MobileNetV3 계열도 사전학습 동작을 유지한 채 export된다.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Hardswish):
            setattr(module, name, _HardSwish())
        elif isinstance(child, nn.Hardsigmoid):
            setattr(module, name, _HardSigmoid())
        elif type(child).__name__ == "SqueezeExcite" and hasattr(child, "conv_reduce"):
            safe = _SafeSqueezeExcite(child)
            make_export_safe(safe)
            setattr(module, name, safe)
        else:
            make_export_safe(child)
    return module


class _StudentV2Base(nn.Module):
    """공통부: timm MobileNet 계열 인코더(stride 4/8/16/32 탭), 전역 문맥 h와 이미지 수준 헤드,
    class_bias·presence 게이트, 전역 색 보정 파라미터."""

    def __init__(self, backbone: str = "mobilenetv2_050", pretrained: bool = True, width: float = 1.0,
                 drop_last_block: bool = False):
        super().__init__()
        import timm  # 지연 import(다른 모델 사용을 막지 않는다)

        self.cfg = cfg = _scaled_config(float(width))
        self.backbone_name = backbone
        self.pretrained = bool(pretrained)
        net = timm.create_model(backbone, pretrained=bool(pretrained))
        if not (hasattr(net, "conv_stem") and hasattr(net, "blocks")):
            raise ValueError(f"{backbone!r} has no conv_stem/blocks (MobileNet/EfficientNet family needed)")
        blocks = list(net.blocks.children())
        if drop_last_block:
            blocks = blocks[:-1]
        self.enc_stem = make_export_safe(nn.Sequential(net.conv_stem, net.bn1))
        self.enc_blocks = make_export_safe(nn.ModuleList(blocks))
        # stride 4/8/16/32 각각의 마지막 블록 위치·채널을 더미 입력으로 찾는다(eval: BN 통계 보존).
        taps = {}
        self.enc_stem.eval()
        self.enc_blocks.eval()
        with torch.no_grad():
            y = self.enc_stem(torch.zeros(1, 3, 64, 64))
            for i, block in enumerate(self.enc_blocks):
                y = block(y)
                taps[64 // int(y.shape[-1])] = (i, int(y.shape[1]))
        self.enc_stem.train()
        self.enc_blocks.train()
        if any(s not in taps for s in (4, 8, 16, 32)):
            raise ValueError(f"{backbone!r} does not expose stride 4/8/16/32 features: {sorted(taps)}")
        self._taps = tuple(taps[s][0] for s in (4, 8, 16, 32))
        self._enc_ch = [taps[s][1] for s in (4, 8, 16, 32)]

        c4 = self._enc_ch[3]
        self.gctx = GlobalContext(c4)
        self.ctx_fc = nn.Sequential(nn.Conv2d(c4, cfg.ctx_hidden, 1), nn.ReLU(inplace=True))
        self.presence_head = nn.Conv2d(cfg.ctx_hidden, cfg.n_classes, 1)
        self.medium_head = nn.Conv2d(cfg.ctx_hidden, cfg.n_water + 1, 1)
        self.restore_global = _zero_conv(cfg.ctx_hidden, 3 + 3 + 9 + 3 * cfg.tone_knots)
        self.class_bias = nn.Parameter(torch.zeros(cfg.n_classes))
        self.presence_gain = nn.Parameter(torch.zeros(1, cfg.n_classes, 1, 1))

    def encode(self, x):
        y = self.enc_stem(x)
        feats = []
        for i, block in enumerate(self.enc_blocks):
            y = block(y)
            if i in self._taps:
                feats.append(y)
        return feats  # s1(H/4), s2(H/8), s3(H/16), s4(H/32)

    def context(self, s4):
        s4 = self.gctx(s4, gap_matmul(s4))
        h = self.ctx_fc(gap_matmul(s4))
        return s4, h, self.presence_head(h), self.medium_head(h), self.restore_global(h)

    def _seg_out(self, feat, presence, size):
        logits = (self.seg_head(feat) + self.class_bias.reshape(1, -1, 1, 1)
                  + self.presence_gain * torch.sigmoid(presence))
        return _resize(logits, size)

    def pack(self, restored, seg_logits, presence, medium):
        n_water = self.cfg.n_water
        return {"restored_image": torch.clamp(restored, 0.0, 1.0), "seg_logits": seg_logits,
                "presence_logits": presence.flatten(1),
                "water_logits": medium[:, :n_water].flatten(1),
                "depth": medium[:, n_water:].flatten(1)}

    def encoder_parameters(self):
        """사전학습 인코더 파라미터(train.py가 낮은 lr 그룹으로 묶는다)."""
        return list(self.enc_stem.parameters()) + list(self.enc_blocks.parameters())

    # ---- 추론 시 입력 저역통과(하네스 리사이즈 앨리어싱 헤지, outputs/analysis_rareclass 2026-10-27) ---- #
    def set_input_blur(self, sigma: float):
        """인코더·디코더 입력에 고정 5×5 가우시안(σ, zero pad, 채널별)을 건다. 복원 경로(전역 색 보정·refiner)는 원 입력.
        재학습 없음. export 시 Conv 1개(+75 상수, +0.16 GF @1024²). sigma<=0이면 해제. 상태 사전에는 저장하지 않는다."""
        sigma = float(sigma)
        if hasattr(self, "input_blur_kernel"):
            del self.input_blur_kernel
        self.input_blur_sigma = 0.0
        if sigma <= 0:
            return self
        g = torch.exp(-torch.arange(-2, 3, dtype=torch.float32) ** 2 / (2 * sigma * sigma))
        g = g / g.sum()
        kernel = (g[:, None] * g[None, :]).expand(3, 1, 5, 5).contiguous().to(next(self.parameters()).device)
        # nn.Parameter(고정)로 둬야 TorchScript export가 initializer로 내보낸다(버퍼는 Constant로 접혀 검사기가 못 셈).
        # load_state_dict는 set_input_blur 전에 끝나므로 기존 ckpt와 충돌하지 않는다.
        self.input_blur_kernel = nn.Parameter(kernel, requires_grad=False)
        self.input_blur_sigma = sigma
        return self

    def _blurred(self, x):
        if getattr(self, "input_blur_sigma", 0.0) > 0:
            # .to(dtype) 같은 Cast를 넣지 않는다: export 시 상수 접힘으로 커널이 initializer가 아닌 Constant가 되어 주최측 검사기가 Conv를 못 센다
            return F.conv2d(x, self.input_blur_kernel, padding=2, groups=3)
        return x

    def switch_to_deploy(self):
        """RepConv를 단일 3x3으로 합친다(export 직전에 호출, 되돌릴 수 없음)."""
        for module in self.modules():
            if isinstance(module, RepConv):
                module.switch_to_deploy()

    def forward_export(self, x):
        outputs = self.forward(x)
        return outputs["restored_image"], outputs["seg_logits"]


class StudentDSFPN(_StudentV2Base):
    """설계 1: DS-FPN(PPM + stride 16/8/4/2 DS-Block×2) → H/2 분할 헤드, H/2 특징 → PixelShuffle 복원 잔차.

    s4 → PPM(1/2/3/6, 각 32ch) → 64
      ↑ + lat(s3) → DS×2(64) [FiLM]  →  ↑ + lat(s2) → DS×2(48) [FiLM]
      ↑ + lat(s1) → DS×2(32)          →  ↑ + s0(24, H/2) → DS×2(24) = f2
    seg = seg_head(f2) + class_bias + presence 게이트 → ×2
    y = base(x) + PixelShuffle(1x1→12 (f2));  y += RepConv(6→12)·RepConv(12→3)([x, y]) → clamp
    """

    def __init__(self, backbone: str = "mobilenetv2_050", pretrained: bool = True, width: float = 1.0,
                 dec_widths: Sequence[int] = (64, 48, 32, 24), ppm_ch: int = 32,
                 ppm_bins: Sequence[int] = (1, 2, 3, 6), drop_last_block: bool = False,
                 refiner_ch: int = 12, refiner_depth: int = 1):
        super().__init__(backbone, pretrained, width, drop_last_block)
        cfg = self.cfg
        c1, c2, c3, c4 = self._enc_ch
        w16, w8, w4, w2 = (int(v) for v in dec_widths)
        self.dec_widths = (w16, w8, w4, w2)
        self.pre_stem = ConvBNAct(3, cfg.stem, 3, 2)
        self.ppm = PPM(c4, ppm_ch, ppm_bins, w16)
        self.lat16 = ConvBNAct(c3, w16, 1, act=False)
        self.blk16 = nn.Sequential(DSBlock(w16), DSBlock(w16))
        self.film16 = _zero_conv(cfg.ctx_hidden, 2 * w16)
        self.proj8 = ConvBNAct(w16, w8, 1, act=False)
        self.lat8 = ConvBNAct(c2, w8, 1, act=False)
        self.blk8 = nn.Sequential(DSBlock(w8), DSBlock(w8))
        self.film8 = _zero_conv(cfg.ctx_hidden, 2 * w8)
        self.proj4 = ConvBNAct(w8, w4, 1, act=False)
        self.lat4 = ConvBNAct(c1, w4, 1, act=False)
        self.blk4 = nn.Sequential(DSBlock(w4), DSBlock(w4))
        self.proj2 = ConvBNAct(w4, w2, 1, act=False)
        self.lat2 = ConvBNAct(cfg.stem, w2, 1, act=False) if cfg.stem != w2 else nn.Identity()
        self.blk2 = nn.Sequential(DSBlock(w2), DSBlock(w2))
        self.seg_head = nn.Conv2d(w2, cfg.n_classes, 1)
        self.res_head = _zero_conv(w2, 12)
        # 전해상도 복원 정제기: 기본 (12ch, 1단) = 기존 그래프. refiner_ch·refiner_depth로 넓히면 FLOPs만 늘고 params는 수 K.
        rc, rd = int(refiner_ch), int(refiner_depth)
        layers = [RepConv(6, rc)] + [RepConv(rc, rc) for _ in range(rd - 1)]
        self.refiner = nn.Sequential(*layers, RepConv(rc, 3, act=False, zero_init=True))

    def decode(self, s0, s1, s2, s3, s4, h):
        p = self.ppm(s4)
        p = _film(self.blk16(_resize(p, s3) + self.lat16(s3)), self.film16(h))
        p = _film(self.blk8(_resize(self.proj8(p), s2) + self.lat8(s2)), self.film8(h))
        p = self.blk4(_resize(self.proj4(p), s1) + self.lat4(s1))
        return self.blk2(_resize(self.proj2(p), s0) + self.lat2(s0))

    def restore_full(self, x, base, f2):
        y = base + pixel_shuffle2(self.res_head(f2))
        return y + self.refiner(torch.cat([x, y], dim=1))

    def forward(self, x):
        xb = self._blurred(x)                      # 분할 경로만 저역통과(기본은 항등)
        s1, s2, s3, s4 = self.encode(xb)
        s4, h, presence, medium, glob = self.context(s4)
        f2 = self.decode(self.pre_stem(xb), s1, s2, s3, s4, h)
        seg = self._seg_out(f2, presence, x.shape[-2:])
        base = global_correct(x, glob, self.cfg.tone_knots)
        return self.pack(self.restore_full(x, base, f2), seg, presence, medium)


class StudentBiSe(_StudentV2Base):
    """설계 2: Detail 분기(RepVGG, 원 해상도부터) + Context 분기(사전학습 인코더) → H/8 BGA-lite 융합.

    Detail: x → 8 @H → 16 @H/2 ×2 → 32 @H/4 ×2 → 64 @H/8 ×2 (RepConv, 배포 시 단일 3x3)
    Context: s4 → PPM → 64 → ↑ + lat(s3) → DS(64) [FiLM] → ↑ + lat(s2) → DS(64) [FiLM] = c8 (H/8)
    융합: f = d8·σ(DW·PW(c8)) + c8·σ(DW·PW(d8)) → DS×2(64) → ↑ + d4(32) → ↑ + d2(16) → seg_head @H/2 → ×2
    복원: base(x) + PixelShuffle(1x1→12 (FiLM(d2))) + 1x1→3 (d_full) → clamp
    fuse_blocks=1이면 ↑+d4, ↑+d2 합 뒤에 DS-Block 하나씩을 둔다(원 설계에 없는 소폭 추가, 0이면 합만).
    """

    def __init__(self, backbone: str = "mobilenetv2_050", pretrained: bool = True, width: float = 1.0,
                 detail_widths: Sequence[int] = (8, 16, 32, 64), ctx_ch: int = 64, ppm_ch: int = 32,
                 ppm_bins: Sequence[int] = (1, 2, 3, 6), fuse_blocks: int = 1):
        super().__init__(backbone, pretrained, width, False)
        cfg = self.cfg
        _, c2, c3, c4 = self._enc_ch
        d1, d2, d4, d8 = (int(v) for v in detail_widths)
        if d8 != ctx_ch:
            raise ValueError(f"detail H/8 width {d8} must equal ctx_ch {ctx_ch} for BGA-lite")
        self.detail_widths = (d1, d2, d4, d8)
        self.det_full = RepConv(3, d1)
        self.det_2 = nn.Sequential(RepConv(d1, d2, 2), RepConv(d2, d2))
        self.det_4 = nn.Sequential(RepConv(d2, d4, 2), RepConv(d4, d4))
        self.det_8 = nn.Sequential(RepConv(d4, d8, 2), RepConv(d8, d8))
        self.ppm = PPM(c4, ppm_ch, ppm_bins, ctx_ch)
        self.lat16 = ConvBNAct(c3, ctx_ch, 1, act=False)
        self.blk16 = DSBlock(ctx_ch)
        self.film16 = _zero_conv(cfg.ctx_hidden, 2 * ctx_ch)
        self.lat8 = ConvBNAct(c2, ctx_ch, 1, act=False)
        self.blk8 = DSBlock(ctx_ch)
        self.film8 = _zero_conv(cfg.ctx_hidden, 2 * ctx_ch)
        self.gate_d = nn.Sequential(ConvBNAct(ctx_ch, ctx_ch, 3, groups=ctx_ch), nn.Conv2d(ctx_ch, d8, 1))
        self.gate_c = nn.Sequential(ConvBNAct(d8, d8, 3, groups=d8), nn.Conv2d(d8, ctx_ch, 1))
        self.blk_f = nn.Sequential(DSBlock(ctx_ch), DSBlock(ctx_ch))
        self.proj4 = ConvBNAct(ctx_ch, d4, 1, act=False)
        self.mix4 = DSBlock(d4) if fuse_blocks else nn.Identity()
        self.proj2 = ConvBNAct(d4, d2, 1, act=False)
        self.mix2 = DSBlock(d2) if fuse_blocks else nn.Identity()
        self.seg_head = nn.Conv2d(d2, cfg.n_classes, 1)
        self.film_r = _zero_conv(cfg.ctx_hidden, 2 * d2)
        self.res_half = _zero_conv(d2, 12)
        self.res_full = _zero_conv(d1, 3)

    def forward(self, x):
        d_full = self.det_full(x)
        d_2 = self.det_2(d_full)
        d_4 = self.det_4(d_2)
        d_8 = self.det_8(d_4)
        _, s2, s3, s4 = self.encode(x)
        s4, h, presence, medium, glob = self.context(s4)
        c = self.ppm(s4)
        c = _film(self.blk16(_resize(c, s3) + self.lat16(s3)), self.film16(h))
        c = _film(self.blk8(_resize(c, s2) + self.lat8(s2)), self.film8(h))
        f = d_8 * torch.sigmoid(self.gate_d(c)) + c * torch.sigmoid(self.gate_c(d_8))
        f = self.blk_f(f)
        f = self.mix4(_resize(self.proj4(f), d_4) + d_4)
        f = self.mix2(_resize(self.proj2(f), d_2) + d_2)
        seg = self._seg_out(f, presence, x.shape[-2:])
        base = global_correct(x, glob, self.cfg.tone_knots)
        y = (base + pixel_shuffle2(self.res_half(_film(d_2, self.film_r(h))))
             + self.res_full(d_full))
        return self.pack(y, seg, presence, medium)


class StudentAttnIter(StudentDSFPN):
    """설계 3: 인코더 마지막 블록(80→160) 제거 + 병목 어텐션 1블록, 분할은 설계 1 DS-FPN,
    복원은 FiLM(h) 조절 DS-Block(24ch)을 가중치 공유로 iters번 펼친 반복 정제.

    s4'(80 @H/32) → LiteAttention(heads 4, CPE, MLP 80→160→80) → PPM/GAP → h
    복원: y₀ = base(x)↓2;  k=1..N: hid = FiLM(DS(1x1([x↓2, y_k, f2]))), y_{k+1} = y_k + 1x1→3(hid)
          out = base(x) + PixelShuffle(1x1→12 (hid_N));  out += RepConv refiner([x, out]) → clamp
    """

    def __init__(self, backbone: str = "mobilenetv2_050", pretrained: bool = True, width: float = 1.0,
                 dec_widths: Sequence[int] = (64, 48, 32, 24), ppm_ch: int = 32,
                 ppm_bins: Sequence[int] = (1, 2, 3, 6), heads: int = 4, mlp_ratio: float = 2.0,
                 attn: str = "relu_linear", iters: int = 3):
        super().__init__(backbone, pretrained, width, dec_widths, ppm_ch, ppm_bins, drop_last_block=True)
        cfg = self.cfg
        w2 = self.dec_widths[3]
        self.attn = LiteAttention(self._enc_ch[3], heads, mlp_ratio, attn)
        self.iters = int(iters)
        self.r_in = ConvBNAct(3 + 3 + w2, w2, 1)
        self.r_blk = DSBlock(w2)
        self.r_film = _zero_conv(cfg.ctx_hidden, 2 * w2)
        self.r_delta = _zero_conv(w2, 3)

    def forward(self, x):
        s1, s2, s3, s4 = self.encode(x)
        s4, h, presence, medium, glob = self.context(self.attn(s4))
        f2 = self.decode(self.pre_stem(x), s1, s2, s3, s4, h)
        seg = self._seg_out(f2, presence, x.shape[-2:])
        base = global_correct(x, glob, self.cfg.tone_knots)
        x2, y = _resize(x, f2), _resize(base, f2)
        gb = self.r_film(h)
        hid = f2
        for _ in range(self.iters):
            hid = _film(self.r_blk(self.r_in(torch.cat([x2, y, f2], dim=1))), gb)
            y = y + self.r_delta(hid)
        return self.pack(self.restore_full(x, base, hid), seg, presence, medium)
