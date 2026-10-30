"""Track 1 restoration teacher with a spatial affine color field and full-size refiner."""

import torch
from torch import nn
from torch.nn import functional as F

from .teacher_restore import NAFBlock


class TeacherCellAffine(nn.Module):
    """Predict one RGB affine transform per ``grid_stride`` input pixels.

    The encoder sees a quarter-size image. Two stride-2 convolutions make a
    stride-16 grid by default; interpolation makes the field continuous between
    cells and also supports input sizes that are not multiples of 16.
    """

    def __init__(self, use_meta=True, grid_stride=16, refiner_width=32, refiner_blocks=8, cell_width=32):
        super().__init__()
        if grid_stride < 4 or grid_stride % 4 or (grid_stride // 4).bit_count() != 1:
            raise ValueError("grid_stride must be a power-of-two multiple of 4")
        if refiner_width < 2 or refiner_width % 2 or refiner_blocks < 0:
            raise ValueError("refiner_width must be positive and even; refiner_blocks must be nonnegative")
        self.use_meta = bool(use_meta)
        self.grid_stride = int(grid_stride)
        self.refiner_width = int(refiner_width)
        self.refiner_blocks = int(refiner_blocks)

        width = int(cell_width)
        self.cell_width = width
        self.cell_intro = nn.Conv2d(3, width, 3, padding=1)
        levels = []
        stride = 4
        while stride < grid_stride:
            levels.extend((NAFBlock(width), nn.Conv2d(width, width, 3, stride=2, padding=1)))
            stride *= 2
        self.cell_encoder = nn.Sequential(*levels, NAFBlock(width), NAFBlock(width))
        # Global context: water-type color cast is image-wide, so each cell also sees
        # the image-level mean feature (zero-initialized, starts as a no-op).
        self.global_ctx = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        nn.init.zeros_(self.global_ctx[2].weight)
        nn.init.zeros_(self.global_ctx[2].bias)
        self.meta_film = nn.Linear(11, 2 * width) if self.use_meta else None
        if self.meta_film is not None:
            nn.init.zeros_(self.meta_film.weight)
            nn.init.zeros_(self.meta_film.bias)
        self.affine_head = nn.Conv2d(width, 12, 3, padding=1)
        nn.init.zeros_(self.affine_head.weight)
        nn.init.zeros_(self.affine_head.bias)

        self.refiner_intro = nn.Conv2d(6, refiner_width, 3, padding=1)
        self.refiner = nn.Sequential(*[NAFBlock(refiner_width) for _ in range(refiner_blocks)])
        self.refiner_out = nn.Conv2d(refiner_width, 3, 3, padding=1)
        nn.init.zeros_(self.refiner_out.weight)
        nn.init.zeros_(self.refiner_out.bias)

    def forward(self, inp, meta=None):
        if inp.ndim != 4 or inp.shape[1] != 3:
            raise ValueError("input must have shape (B,3,H,W)")
        if self.use_meta and (meta is None or meta.shape != (inp.shape[0], 11)):
            raise ValueError("use_meta=True requires a (B,11) metadata tensor")
        h, w = inp.shape[-2:]
        small = F.interpolate(inp, size=(max(1, (h + 3) // 4), max(1, (w + 3) // 4)),
                              mode="bilinear", align_corners=False)
        feat = self.cell_encoder(self.cell_intro(small))
        feat = feat + self.global_ctx(feat.mean(dim=(2, 3)))[..., None, None]
        if self.use_meta:
            gamma, beta = self.meta_film(meta.to(device=inp.device, dtype=feat.dtype)).chunk(2, dim=1)
            feat = feat * (1 + gamma[..., None, None]) + beta[..., None, None]

        # Keep the color matrix and interpolation in fp32 under autocast.
        # The zero-initialized head represents M=I and b=0 exactly.
        delta = self.affine_head(feat)
        with torch.autocast(device_type=inp.device.type, enabled=False):
            delta = delta.float()
            delta = F.interpolate(delta, size=(h, w), mode="bilinear", align_corners=False)
            matrix = delta[:, :9].reshape(-1, 3, 3, h, w)
            colors = inp.float()
            affine = (colors + torch.einsum("boihw,bihw->bohw", matrix, colors)
                      + delta[:, 9:])

        residual = self.refiner_out(self.refiner(self.refiner_intro(torch.cat((colors, affine), dim=1))))
        restored = (affine + residual.float()).clamp(0, 1)
        return {"restored_image": restored, "affine_image": affine}

    def config_dict(self):
        return {"name": "teacher_cellaff", "use_meta": self.use_meta,
                "grid_stride": self.grid_stride, "refiner_width": self.refiner_width,
                "refiner_blocks": self.refiner_blocks, "cell_width": self.cell_width}
