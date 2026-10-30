"""Track 1 physics-constrained restoration teacher (method candidate M1).

The RSUIGM forward model is ``I_c = A_c exp(-beta_c z) J_c + B_c (1-exp(-beta_c z))``.
Our analysis shows the channel constants ``(A_c, B_c, beta_c)`` are almost fully
determined by the water type ``w`` and depth ``d`` (per-water K_d IQR <= 0.004),
while the spatial variation is explained by a single relative-distance map ``z``
(32x32-patch z reconstruction 46 dB; 32x24 oracle z map 32.9 dB).

So instead of the free per-cell affine field of :class:`TeacherCellAffine`, this
model:
  1. Looks up ``(A, B, beta)`` from a ``(water, depth)`` table (meta mode) or
     regresses them from a global feature (nometa mode, for Track 2 student
     transfer / ablation), optionally adding a small learned correction.
  2. Predicts a low-resolution ``z`` grid with the same quarter-size NAFBlock
     encoder as the cell-affine teacher and upsamples it.
  3. Analytically inverts the forward model to get ``J0``.
  4. Refines ``J0`` with a small NAFBlock residual stack.

The lookup table is copied into a buffer at construction so it rides inside the
checkpoint; the server never needs the fit jsonl to load the model.
"""

import torch
from torch import nn
from torch.nn import functional as F

from ..data.rsuigm_table import (
    RSUIGMTable, N_WATER, N_DEPTH, N_CHANNEL, A_RANGE, B_RANGE, BETA_MIN,
    depth_index, _PLACEHOLDER, _PLACEHOLDER_ZMEAN,
)
from .teacher_restore import NAFBlock

_BLACK_THRESH = 2.0 / 255.0


def _inv_softplus(y):
    """Inverse of ``softplus`` for strictly-positive ``y`` (for bias init)."""
    y = torch.as_tensor(y, dtype=torch.float32).clamp(min=1e-4)
    return y + torch.log(-torch.expm1(-y))


def physical_inverse(image, a, b, beta, zmap, black_thresh=_BLACK_THRESH):
    """Invert the RSUIGM forward model to recover ``J0`` from a degraded image.

    ``t = exp(-beta z)``; ``J0 = (I - B (1 - t)) / clamp(A t, min=0.02)`` then
    clamped to ``[0, 1]``. Fully-black inputs (``max < black_thresh``) pass
    through unchanged. Runs in fp32 regardless of autocast.

    Shapes: ``image`` (N,3,H,W); ``a``/``b``/``beta`` (N,3); ``zmap`` (N,1,H,W)
    or (N,H,W).
    """
    image = image.float()
    a = a.float()[:, :, None, None]
    b = b.float()[:, :, None, None]
    beta = beta.float()[:, :, None, None]
    z = zmap.float()
    if z.ndim == 3:
        z = z[:, None]
    t = torch.exp(-beta * z)
    j0 = (image - b * (1.0 - t)) / torch.clamp(a * t, min=0.02)
    j0 = j0.clamp(0.0, 1.0)
    black = image.amax(dim=(1, 2, 3), keepdim=True) < black_thresh
    return torch.where(black, image, j0)


class TeacherPhysRestore(nn.Module):
    """Physics-constrained restoration teacher (constant lookup + z map + inverse)."""

    def __init__(self, use_meta=True, grid_stride=16, cell_width=64, refiner_width=32,
                 refiner_blocks=8, learn_const_delta=True,
                 fit_jsonl=None, summary_json=None, allow_placeholder=False,
                 allowed_scenes=None):
        super().__init__()
        if grid_stride < 4 or grid_stride % 4 or (grid_stride // 4).bit_count() != 1:
            raise ValueError("grid_stride must be a power-of-two multiple of 4")
        if refiner_width < 2 or refiner_width % 2 or refiner_blocks < 0:
            raise ValueError("refiner_width must be positive and even; refiner_blocks must be nonnegative")
        self.use_meta = bool(use_meta)
        self.grid_stride = int(grid_stride)
        self.refiner_width = int(refiner_width)
        self.refiner_blocks = int(refiner_blocks)
        self.learn_const_delta = bool(learn_const_delta)

        # Lookup table → buffers (so the checkpoint is self-contained).
        #
        # New training (``allow_placeholder=False``, the default) must build the
        # table from the real analysis products; RSUIGMTable(require=True) raises
        # if the fit/summary files are missing or empty, so a run never starts
        # silently on placeholder physics constants.
        #
        # The checkpoint-load path (``load_teacher_restore`` sets
        # ``allow_placeholder=True``) has no fit jsonl on the server, so it falls
        # back to placeholder buffers that ``load_state_dict`` immediately
        # overwrites with the checkpoint's own table.
        table_kwargs = {}
        if fit_jsonl is not None:
            table_kwargs["fit_jsonl"] = fit_jsonl
        if summary_json is not None:
            table_kwargs["summary_json"] = summary_json
        if allowed_scenes is not None:
            table_kwargs["allowed_scenes"] = allowed_scenes
        try:
            table = RSUIGMTable(require=not allow_placeholder, **table_kwargs)
            table_buf = torch.from_numpy(table.table).clone()
            z_mean = torch.from_numpy(table.z_mean).clone()
            bounds_lo = torch.from_numpy(table.bounds_lo).clone()
            bounds_hi = torch.from_numpy(table.bounds_hi).clone()
        except Exception:
            if not allow_placeholder:
                # New-training path: do not mask a missing/invalid physics table.
                raise
            # Checkpoint-load path only: placeholder buffers, overwritten by load.
            table_buf = _PLACEHOLDER[None, None].repeat(N_WATER * N_DEPTH, 0).reshape(
                N_WATER, N_DEPTH, N_CHANNEL)
            table_buf = torch.from_numpy(table_buf).clone()
            z_mean = torch.full((N_WATER,), float(_PLACEHOLDER_ZMEAN))
            lo = _PLACEHOLDER.copy(); hi = _PLACEHOLDER.copy() + 1e-2
            bounds_lo = torch.from_numpy(lo).clone()
            bounds_hi = torch.from_numpy(hi).clone()
        self.register_buffer("const_table", table_buf)
        self.register_buffer("z_mean_table", z_mean)
        self.register_buffer("bounds_lo", bounds_lo)
        self.register_buffer("bounds_hi", bounds_hi)

        width = int(cell_width)
        self.cell_width = width
        self.cell_intro = nn.Conv2d(3, width, 3, padding=1)
        levels = []
        stride = 4
        while stride < grid_stride:
            levels.extend((NAFBlock(width), nn.Conv2d(width, width, 3, stride=2, padding=1)))
            stride *= 2
        self.cell_encoder = nn.Sequential(*levels, NAFBlock(width), NAFBlock(width))
        # Global context: the water-type color cast is image-wide.
        self.global_ctx = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        nn.init.zeros_(self.global_ctx[2].weight)
        nn.init.zeros_(self.global_ctx[2].bias)
        self.meta_film = nn.Linear(11, 2 * width) if self.use_meta else None
        if self.meta_film is not None:
            nn.init.zeros_(self.meta_film.weight)
            nn.init.zeros_(self.meta_film.bias)

        # z grid head (1 channel, softplus>=0). Zero-init so the grid starts flat
        # at the per-water z_mean (added as a bias inside forward).
        self.z_head = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.z_head.weight)
        nn.init.zeros_(self.z_head.bias)
        # nometa z base: learnable scalar initialised at the global median z_mean.
        global_z = float(z_mean.median())
        self.z_base = nn.Parameter(_inv_softplus(torch.tensor(global_z)).reshape(()))

        # Constant heads.
        if self.use_meta:
            if self.learn_const_delta:
                # meta + global feature -> 9-channel multiplicative correction,
                # zero-initialised, bounded to +-20% via tanh.
                self.const_delta = nn.Sequential(
                    nn.Linear(11 + width, width), nn.GELU(), nn.Linear(width, N_CHANNEL))
                nn.init.zeros_(self.const_delta[2].weight)
                nn.init.zeros_(self.const_delta[2].bias)
            else:
                self.const_delta = None
        else:
            # Regress A/B/beta from the global feature, bounded to the full-table
            # range via sigmoid.
            #
            # The bias is initialised so the starting constants equal the
            # global-median table value (weight zero -> output is bias-only).
            # A plain zero bias gave sigmoid(0)=0.5, i.e. the *midpoint of the
            # full (water,depth) range*. Because beta's range reaches the
            # deep-water maximum, that midpoint beta is large: for a dark valid
            # condition (e.g. w=9, d=50) t=exp(-beta z)->0, A*t is clamped to
            # 0.02, and J0=(I - B(1-t))/0.02 fell below 0 for every pixel, so
            # clamp(0,1) saturated and the const/z heads received no gradient.
            # Starting at the realistic median keeps beta/B moderate so J0 lands
            # in the open interval and gradients flow from step one.
            self.const_head = nn.Linear(width, N_CHANNEL)
            nn.init.zeros_(self.const_head.weight)
            with torch.no_grad():
                med = self.const_table.reshape(-1, N_CHANNEL).median(dim=0).values.float()
                span = (self.bounds_hi - self.bounds_lo).float().clamp(min=1e-6)
                frac = ((med - self.bounds_lo.float()) / span).clamp(1e-4, 1 - 1e-4)
                self.const_head.bias.copy_(torch.log(frac / (1 - frac)))

        self.refiner_intro = nn.Conv2d(6, refiner_width, 3, padding=1)
        self.refiner = nn.Sequential(*[NAFBlock(refiner_width) for _ in range(refiner_blocks)])
        self.refiner_out = nn.Conv2d(refiner_width, 3, 3, padding=1)
        nn.init.zeros_(self.refiner_out.weight)
        nn.init.zeros_(self.refiner_out.bias)

    def _constants(self, meta, gfeat, water):
        """Return A, B, beta each shaped (B, 3)."""
        if self.use_meta:
            di = depth_index(meta[:, 10] * 100.0).to(water.device)
            values = self.const_table[water, di]  # (B, 9)
            if self.const_delta is not None:
                delta = 0.2 * torch.tanh(
                    self.const_delta(torch.cat((meta.to(gfeat.dtype), gfeat), dim=1)))
                values = values * (1.0 + delta.to(values.dtype))
        else:
            raw = self.const_head(gfeat)
            values = self.bounds_lo + (self.bounds_hi - self.bounds_lo) * torch.sigmoid(raw.to(self.bounds_lo.dtype))
        a = values[:, 0:3].clamp(*A_RANGE)
        b = values[:, 3:6].clamp(*B_RANGE)
        beta = values[:, 6:9].clamp(min=BETA_MIN)
        return a, b, beta

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
            meta = meta.to(device=inp.device, dtype=feat.dtype)
            gamma, beta_film = self.meta_film(meta).chunk(2, dim=1)
            feat = feat * (1 + gamma[..., None, None]) + beta_film[..., None, None]
            water = meta[:, :10].argmax(dim=1)
        else:
            water = None
        gfeat = feat.mean(dim=(2, 3))

        a, b, beta = self._constants(meta if self.use_meta else None, gfeat, water)
        z_grid = self.z_head(feat)

        # Keep the inverse and interpolation in fp32 under autocast. The zero-init
        # z head + per-water z base makes the initial z grid flat at z_mean.
        with torch.autocast(device_type=inp.device.type, enabled=False):
            z_grid = z_grid.float()
            if self.use_meta:
                z_base = _inv_softplus(self.z_mean_table[water]).to(z_grid.dtype)[:, None, None, None]
            else:
                z_base = self.z_base.float()
            z_grid = F.softplus(z_grid + z_base)
            zmap = F.interpolate(z_grid, size=(h, w), mode="bilinear", align_corners=False)
            j0 = physical_inverse(inp, a, b, beta, zmap)

        residual = self.refiner_out(self.refiner(self.refiner_intro(torch.cat((inp, j0.to(inp.dtype)), dim=1))))
        restored = (j0 + residual.float()).clamp(0, 1)
        return {"restored_image": restored, "affine_image": j0, "zmap": zmap,
                "constants": {"A": a, "B": b, "beta": beta}}

    def config_dict(self):
        return {"name": "teacher_phys", "use_meta": self.use_meta,
                "grid_stride": self.grid_stride, "cell_width": self.cell_width,
                "refiner_width": self.refiner_width, "refiner_blocks": self.refiner_blocks,
                "learn_const_delta": self.learn_const_delta}
