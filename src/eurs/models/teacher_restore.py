"""Track 1 복원 교사: NAFNet(Chen et al. ECCV 2022)의 순수 torch 자체 구현.

외부 레포를 복사하지 않고 논문 구조(LayerNorm2d, SimpleGate, Simplified Channel
Attention, NAFBlock, U자형 인코더-디코더)를 직접 구현한다. 기본은 입력 3ch → 잔차 3ch,
전역 분기를 켜면 전체 영상의 색·저주파 맵을 먼저 보정한 뒤 NAFNet 잔차를 더한다.
GAN·perceptual 손실은 쓰지 않는다.
"""
import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    """채널축 LayerNorm(공간 위치마다 채널 통계로 정규화)."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        # fp16 AMP에서 분산 계산이 overflow하면 GradScaler가 step을 계속 건너뛰어 출력이 상수로
        # 무너진다(2026-09-30 bengio 재현: loss 0.806 고정, val PSNR 9.1). 통계는 fp32로 계산한다.
        dtype = x.dtype
        x = x.float()
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).square().mean(dim=1, keepdim=True)
        x = ((x - mean) / torch.sqrt(var + self.eps)).to(dtype)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, channels, dw_expand=2, ffn_expand=2, drop_out=0.0):
        super().__init__()
        dw = channels * dw_expand
        self.conv1 = nn.Conv2d(channels, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.conv3 = nn.Conv2d(dw // 2, channels, 1)
        # Simplified Channel Attention: GAP → 1x1 → 채널 스케일.
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.gate = SimpleGate()
        ffn = channels * ffn_expand
        self.conv4 = nn.Conv2d(channels, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, channels, 1)
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.drop1 = nn.Dropout2d(drop_out) if drop_out > 0 else nn.Identity()
        self.drop2 = nn.Dropout2d(drop_out) if drop_out > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gate(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + self.drop1(x) * self.beta
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.gate(x)
        x = self.conv5(x)
        return y + self.drop2(x) * self.gamma


def metadata_vector(water, depth, device=None):
    """파일명 수질(0..9)과 깊이(미터)를 11차원 Track 1 메타데이터로 변환."""
    water = torch.as_tensor(water, device=device, dtype=torch.long).reshape(-1)
    depth = torch.as_tensor(depth, device=device, dtype=torch.float32).reshape(-1)
    if water.numel() != depth.numel() or torch.any((water < 0) | (water > 9)):
        raise ValueError("water must be 0..9 and have the same batch size as depth")
    return torch.cat((F.one_hot(water, 10).float(), (depth / 100.0)[:, None]), dim=1)


class GlobalCorrection(nn.Module):
    """전체 영상의 색 통계에서 전역 affine와 8x16 저주파 보정 맵을 추정."""

    def __init__(self, use_meta=False):
        super().__init__()
        self.use_meta = use_meta
        layers = []
        channels = (3, 32, 64, 128, 192, 256)
        for cin, cout in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
        self.encoder = nn.Sequential(*layers)
        self.mlp = nn.Sequential(nn.Linear(256 + (11 if use_meta else 0), 256), nn.ReLU(inplace=True))
        self.affine = nn.Linear(256, 6)
        self.maps = nn.Linear(256, 6 * 8 * 16)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)
        nn.init.zeros_(self.maps.weight)
        nn.init.zeros_(self.maps.bias)

    def forward(self, x, meta=None):
        if self.use_meta and (meta is None or meta.shape != (x.shape[0], 11)):
            raise ValueError("use_meta=True requires a (B,11) metadata tensor")
        h, w = x.shape[-2:]
        # 긴 변/짧은 변의 비율을 유지하며 128x256 캔버스에 맞춘다.
        scale = min(128 / h, 256 / w)
        rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
        small = F.interpolate(x, size=(rh, rw), mode="bilinear", align_corners=False)
        small = F.pad(small, (0, 256 - rw, 0, 128 - rh), mode="replicate")
        feat = self.encoder(small).mean(dim=(-2, -1))
        if self.use_meta:
            feat = torch.cat((feat, meta.to(device=x.device, dtype=x.dtype)), dim=1)
        feat = self.mlp(feat)
        gain_offset = self.affine(feat)
        gain = 1 + gain_offset[:, :3, None, None]
        offset = gain_offset[:, 3:, None, None]
        maps = self.maps(feat).reshape(-1, 6, 8, 16)
        maps = F.interpolate(maps, size=(h, w), mode="bilinear", align_corners=False)
        return (x * (gain * (1 + maps[:, :3])) + offset + maps[:, 3:]).clamp(0, 1)


class TeacherRestore(nn.Module):
    """NAFNet U-Net. 기본 width 32, enc [1,1,1,28], middle 1, dec [1,1,1,1] (약 17M).

    config로 폭·블록 수를 축소할 수 있다. 입력은 padder_size(=2^깊이) 배수로 패딩 후 잘라낸다.
    """

    def __init__(self, width=32, enc_blk_nums=(1, 1, 1, 28), middle_blk_num=1,
                 dec_blk_nums=(1, 1, 1, 1), img_channel=3, global_branch=False,
                 restore_input="concat", use_meta=False):
        super().__init__()
        if restore_input not in ("concat", "corrected"):
            raise ValueError("restore_input must be concat or corrected")
        if use_meta and not global_branch:
            raise ValueError("use_meta requires global_branch")
        self.global_branch = global_branch
        self.restore_input = restore_input
        self.use_meta = use_meta
        self.global_correction = GlobalCorrection(use_meta) if global_branch else None
        intro_channels = img_channel * 2 if global_branch and restore_input == "concat" else img_channel
        self.intro = nn.Conv2d(intro_channels, width, 3, padding=1)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2
        self.middle = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
        self.padder_size = 2 ** len(self.encoders)

    def _pad(self, x):
        _, _, h, w = x.shape
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    def forward(self, inp, crop_box=None, meta=None):
        if self.global_branch:
            if isinstance(inp, (list, tuple)):
                if crop_box is None or len(inp) != len(crop_box):
                    raise ValueError("a crop box is required for every full image")
                corrected = []
                cropped = []
                for i, full in enumerate(inp):
                    if full.ndim == 3:
                        full = full[None]
                    one_meta = meta[i:i + 1] if meta is not None else None
                    y0 = self.global_correction(full, one_meta)
                    x0, y, x1, y1 = [int(v) for v in crop_box[i]]
                    corrected.append(y0[..., y:y1, x0:x1])
                    cropped.append(full[..., y:y1, x0:x1])
                base = torch.cat(corrected, dim=0)
                x = torch.cat(cropped, dim=0)
            else:
                base = self.global_correction(inp, meta)
                if crop_box is None:
                    x = inp
                else:
                    if inp.shape[0] != 1:
                        raise ValueError("tensor crop_box requires batch size 1; use a list for batched crops")
                    x0, y, x1, y1 = [int(v) for v in crop_box]
                    base, x = base[..., y:y1, x0:x1], inp[..., y:y1, x0:x1]
            naf_input = torch.cat((x, base), dim=1) if self.restore_input == "concat" else base
        else:
            if crop_box is not None or isinstance(inp, (list, tuple)):
                raise ValueError("crop_box/full image lists require global_branch")
            base = inp
            naf_input = inp
        h, w = base.shape[-2:]
        x = self._pad(naf_input)
        feat = self.intro(x)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            feat = encoder(feat)
            skips.append(feat)
            feat = down(feat)
        feat = self.middle(feat)
        for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            feat = up(feat)
            feat = feat + skip
            feat = decoder(feat)
        out = self.ending(feat)[..., :h, :w]
        restored = torch.clamp(base + out, 0.0, 1.0)
        result = {"restored_image": restored}
        if self.global_branch:
            result["corrected_image"] = base
        return result

    def config_dict(self):
        return {"name": "teacher_restore", "global_branch": self.global_branch,
                "restore_input": self.restore_input, "use_meta": self.use_meta}


def load_teacher_restore(ckpt, device="cpu", **kwargs):
    """체크포인트 config에서 복원 교사 종류와 크기를 복원한다."""
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    model_cfg = checkpoint.get("config", {}).get("model", {})
    if model_cfg.get("name") == "teacher_cellaff":
        from .teacher_cellaff import TeacherCellAffine
        build = {k: model_cfg[k] for k in ("use_meta", "grid_stride", "refiner_width", "refiner_blocks", "cell_width")
                 if k in model_cfg}
        build.update(kwargs)
        model = TeacherCellAffine(**build)
        model.load_state_dict(checkpoint.get("model", checkpoint))
        return model.to(device).eval()
    if model_cfg.get("name") == "teacher_phys":
        from .teacher_phys import TeacherPhysRestore
        build = {k: model_cfg[k] for k in ("use_meta", "grid_stride", "cell_width",
                                           "refiner_width", "refiner_blocks", "learn_const_delta")
                 if k in model_cfg}
        # Loading a checkpoint: the fit/summary files need not exist here, the
        # table buffer is restored from the state dict. Allow the placeholder
        # init unless the caller explicitly overrides it.
        build.setdefault("allow_placeholder", True)
        build.update(kwargs)
        model = TeacherPhysRestore(**build)
        model.load_state_dict(checkpoint.get("model", checkpoint))
        return model.to(device).eval()
    if model_cfg.get("name", "teacher_restore") != "teacher_restore":
        raise ValueError(f"Unknown restoration teacher: {model_cfg['name']}")
    build = {k: model_cfg[k] for k in ("width", "enc_blk_nums", "middle_blk_num", "dec_blk_nums",
                                      "global_branch", "restore_input", "use_meta")
             if k in model_cfg}
    build.update(kwargs)
    if "enc_blk_nums" in build:
        build["enc_blk_nums"] = tuple(build["enc_blk_nums"])
    if "dec_blk_nums" in build:
        build["dec_blk_nums"] = tuple(build["dec_blk_nums"])
    model = TeacherRestore(**build)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    return model.to(device).eval()
