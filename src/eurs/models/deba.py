"""DeBA-style adapters (approximate reproduction of the CVPR 2026 submission).

The paper uses DCNv4. This implementation uses torchvision's modulated DCNv2
operator with sigmoid masks; its offset/mask predictor is depthwise separable.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import deform_conv2d


class DCM(nn.Module):
    """Shared deformable convolution module with four offset groups by default."""

    def __init__(self, d_b=64, groups=4, kernel=3):
        super().__init__()
        if d_b % groups or kernel % 2 != 1:
            raise ValueError("d_b must be divisible by groups and kernel must be odd")
        self.groups = groups
        self.kernel = kernel
        self.input_proj = nn.Conv2d(d_b, d_b, 1)
        self.offset_depthwise = nn.Conv2d(d_b, d_b, kernel, padding=kernel // 2, groups=d_b)
        self.offset_pointwise = nn.Conv2d(d_b, groups * kernel * kernel * 3, 1)
        self.weight = nn.Parameter(torch.empty(d_b, d_b // groups, kernel, kernel))
        self.bias = nn.Parameter(torch.zeros(d_b))
        self.output_proj = nn.Conv2d(d_b, d_b, 1)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        # Zero offsets start at the regular 3x3 sampling locations. The sigmoid
        # modulation starts at 0.5; the enclosing zero-up adapter is identity.
        nn.init.zeros_(self.offset_pointwise.weight)
        nn.init.zeros_(self.offset_pointwise.bias)

    def forward(self, x):
        dtype = x.dtype
        projected = self.input_proj(x)
        prediction = self.offset_pointwise(self.offset_depthwise(x))
        n_offset = self.groups * self.kernel * self.kernel * 2
        offset, mask = prediction[:, :n_offset], prediction[:, n_offset:].sigmoid()
        # torchvision deform_conv2d can reject bf16; keep only this op in fp32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = deform_conv2d(projected.float(), offset.float(), self.weight.float(),
                              self.bias.float(), padding=self.kernel // 2, mask=mask.float())
        return self.output_proj(y.to(dtype))


class DeBA(nn.Module):
    """One shared DCM across BB layers and selected FP levels.

    This is a DeBA-style adapter (approximate reproduction): DCNv2 modulation
    with sigmoid masks replaces the paper's DCNv4 operation. The shared BB
    LayerNorm and per-layer projections/scales follow the paper's Eqs. 3-6.
    """

    def __init__(self, hidden, num_bb_layers, fp_channels, fp_levels, d_b=64):
        super().__init__()
        self.dcm = DCM(d_b=d_b)
        self.norm = nn.LayerNorm(d_b)
        self.bb_down = nn.ModuleList(nn.Linear(hidden, d_b) for _ in range(num_bb_layers))
        self.bb_up = nn.ModuleList(nn.Linear(d_b, hidden) for _ in range(num_bb_layers))
        self.bb_alpha = nn.ParameterList(nn.Parameter(torch.ones(())) for _ in range(num_bb_layers))
        self.fp_levels = tuple(fp_levels)
        self.fp_down = nn.ModuleDict({str(i): nn.Conv2d(fp_channels, d_b, 1) for i in fp_levels})
        self.fp_up = nn.ModuleDict({str(i): nn.Conv2d(d_b, fp_channels, 1) for i in fp_levels})
        self.fp_alpha = nn.ParameterDict({str(i): nn.Parameter(torch.ones(())) for i in fp_levels})
        for up in (*self.bb_up, *self.fp_up.values()):
            nn.init.zeros_(up.weight)
            nn.init.zeros_(up.bias)

    def forward_bb(self, tokens, layer, grid_h, grid_w, prefix):
        patches = tokens[:, prefix:]
        if patches.shape[1] != grid_h * grid_w:
            raise ValueError("patch token count does not match the image grid")
        x = self.bb_down[layer](patches)
        x = x.transpose(1, 2).reshape(tokens.shape[0], -1, grid_h, grid_w)
        x = self.dcm(x).flatten(2).transpose(1, 2)
        residual = self.bb_up[layer](F.gelu(self.norm(x))) * self.bb_alpha[layer]
        return torch.cat((tokens[:, :prefix], patches + residual), dim=1)

    def forward_fp(self, feature, level):
        key = str(level)
        residual = self.fp_up[key](F.gelu(self.dcm(self.fp_down[key](feature))))
        return feature + self.fp_alpha[key] * residual
