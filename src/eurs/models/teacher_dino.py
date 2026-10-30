"""Track 1 DINOv2/DINOv3 ViT 분할 교사."""

import torch
from torch import nn
from torch.nn import functional as F

from eurs.models.deba import DeBA


class _PyramidLevel(nn.Module):
    def __init__(self, hidden, channels, scale):
        super().__init__()
        self.project = nn.Conv2d(hidden, channels, 1)
        if scale == 4:
            self.resize = nn.Sequential(nn.ConvTranspose2d(channels, channels, 2, 2),
                                        nn.GELU(), nn.ConvTranspose2d(channels, channels, 2, 2))
        elif scale == 2:
            self.resize = nn.ConvTranspose2d(channels, channels, 2, 2)
        elif scale == 1:
            self.resize = nn.Identity()
        else:
            self.resize = nn.AvgPool2d(2, 2, ceil_mode=True)
        self.refine = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GELU())

    def forward(self, x, size):
        x = self.resize(self.project(x))
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.refine(x)


class _UperDecoder(nn.Module):
    def __init__(self, channels, num_labels):
        super().__init__()
        self.ppm = nn.ModuleList(nn.Sequential(nn.AdaptiveAvgPool2d(n),
                                                nn.Conv2d(channels, channels, 1), nn.GELU())
                                 for n in (1, 2, 3, 6))
        self.ppm_fuse = nn.Conv2d(channels * 5, channels, 3, padding=1)
        self.lateral = nn.ModuleList(nn.Conv2d(channels, channels, 1) for _ in range(3))
        self.fpn = nn.ModuleList(nn.Conv2d(channels, channels, 3, padding=1) for _ in range(3))
        self.fuse = nn.Sequential(nn.Conv2d(channels * 4, channels, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(channels, num_labels, 1))

    def forward(self, features):
        low = features[-1]
        pooled = [F.interpolate(branch(low), size=low.shape[-2:], mode="bilinear",
                                align_corners=False) for branch in self.ppm]
        top = self.ppm_fuse(torch.cat([low, *pooled], dim=1))
        outputs = [None, None, None, top]
        for i in (2, 1, 0):
            top = self.lateral[i](features[i]) + F.interpolate(
                top, size=features[i].shape[-2:], mode="bilinear", align_corners=False)
            outputs[i] = self.fpn[i](top)
        size = outputs[0].shape[-2:]
        return self.fuse(torch.cat([F.interpolate(x, size=size, mode="bilinear",
                                                 align_corners=False) if x.shape[-2:] != size else x
                                    for x in outputs], dim=1))


class TeacherDino(nn.Module):
    def __init__(self, backbone="facebook/dinov2-large", freeze=True, unfreeze_last=0,
                 layers=None, decoder_ch=256, num_labels=5, pretrained=True,
                 backbone_config=None, input_mode="rgb", adapter=None,
                 adapter_parts="bb+fp", adapter_dim=64, grad_ckpt=False):
        super().__init__()
        if input_mode != "rgb":
            raise ValueError("TeacherDino supports input_mode='rgb' only")
        from transformers import AutoConfig, AutoModel, DINOv3ViTConfig, Dinov2Config
        if backbone_config is not None:
            config_data = dict(backbone_config)
            model_type = config_data.pop("model_type")
            config = AutoConfig.for_model(model_type, **config_data)
            self.encoder = AutoModel.from_config(config)
        elif pretrained:
            self.encoder = AutoModel.from_pretrained(backbone)
        else:
            config = (DINOv3ViTConfig() if "dinov3" in backbone.lower() else Dinov2Config())
            self.encoder = AutoModel.from_config(config)
        config = self.encoder.config
        if config.model_type not in ("dinov2", "dinov3_vit"):
            raise ValueError(f"Unsupported DINO backbone: {config.model_type}")
        self.backbone = backbone
        self.input_mode = input_mode
        self.freeze = bool(freeze)
        self.unfreeze_last = int(unfreeze_last)
        self.patch_size = int(config.patch_size)
        self.num_labels = int(num_labels)
        self.decoder_ch = int(decoder_ch)
        if adapter not in (None, "deba"):
            raise ValueError("adapter must be None or 'deba'")
        if adapter_parts not in ("bb+fp", "bb", "fp"):
            raise ValueError("adapter_parts must be 'bb+fp', 'bb', or 'fp'")
        self.adapter_name = adapter
        self.adapter_parts = adapter_parts
        self.adapter_dim = int(adapter_dim)
        self.grad_ckpt = bool(grad_ckpt)
        depth = int(config.num_hidden_layers)
        self.layers = list(layers) if layers is not None else [depth // 4 - 1, depth // 2 - 1,
                                                                 3 * depth // 4 - 1, depth - 1]
        if len(self.layers) != 4 or any(not 0 <= i < depth for i in self.layers):
            raise ValueError("layers must contain four valid zero-based block indices")
        if not 0 <= self.unfreeze_last <= depth:
            raise ValueError("unfreeze_last must be between 0 and the backbone depth")
        blocks = self.encoder.encoder.layer if config.model_type == "dinov2" else self.encoder.model.layer
        self.encoder.requires_grad_(not self.freeze)
        if self.freeze and self.unfreeze_last:
            for block in blocks[-self.unfreeze_last:]:
                block.requires_grad_(True)
        self.pyramid = nn.ModuleList(_PyramidLevel(config.hidden_size, decoder_ch, scale)
                                     for scale in (4, 2, 1, 0.5))
        self.decoder = _UperDecoder(decoder_ch, num_labels)
        self.presence_head = nn.Linear(config.hidden_size, num_labels)
        self.adapter = (DeBA(config.hidden_size, depth if adapter_parts in ("bb+fp", "bb") else 0,
                             decoder_ch, (0, 1, 3) if adapter_parts in ("bb+fp", "fp") else (),
                             d_b=self.adapter_dim) if adapter == "deba" else None)
        self._adapter_grid = None
        if self.grad_ckpt:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if self.adapter is not None and adapter_parts in ("bb+fp", "bb"):
            for index, block in enumerate(blocks):
                block.register_forward_hook(self._bb_hook(index))
        self.register_buffer("image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
                             persistent=False)
        if self.freeze:
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        self._set_checkpoint_block_mode(mode)
        return self

    def _set_checkpoint_block_mode(self, mode):
        if self.grad_ckpt and mode:
            # HF checkpoints a block only when block.training is true. Keep all
            # child layers in eval mode when the frozen encoder is held in eval.
            blocks = (self.encoder.encoder.layer if self.encoder.config.model_type == "dinov2"
                      else self.encoder.model.layer)
            for block in blocks:
                block.training = True

    def _bb_hook(self, index):
        def hook(_module, _inputs, output):
            grid_h, grid_w = self._adapter_grid
            tokens = output[0] if isinstance(output, tuple) else output
            prefix = tokens.shape[1] - grid_h * grid_w
            if prefix < 1:
                raise ValueError("DINO block has fewer tokens than the patch grid")
            adapted = self.adapter.forward_bb(tokens, index, grid_h, grid_w, prefix)
            return (adapted, *output[1:]) if isinstance(output, tuple) else adapted
        return hook

    def config_dict(self):
        return {"name": "teacher_dino", "backbone": self.backbone, "freeze": self.freeze,
                "unfreeze_last": self.unfreeze_last, "layers": self.layers,
                "decoder_ch": self.decoder_ch, "num_labels": self.num_labels,
                "input_mode": self.input_mode, "backbone_config": self.encoder.config.to_dict(),
                "adapter": self.adapter_name, "adapter_parts": self.adapter_parts,
                "adapter_dim": self.adapter_dim, "grad_ckpt": self.grad_ckpt}

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("TeacherDino rgb expects (B, 3, H, W) input")
        h, w = x.shape[-2:]
        p = self.patch_size
        padded = F.pad(x, (0, (-w) % p, 0, (-h) % p), mode="replicate")
        ph, pw = padded.shape[-2:]
        normalized = (padded - self.image_mean) / self.image_std
        grid_h, grid_w = ph // p, pw // p
        self._adapter_grid = (grid_h, grid_w)
        # Keep the grid through backward: checkpointed blocks rerun their hooks.
        if self.freeze and self.unfreeze_last == 0 and self.adapter is None:
            with torch.no_grad():
                result = self.encoder(normalized, output_hidden_states=True, return_dict=True)
        else:
            result = self.encoder(normalized, output_hidden_states=True, return_dict=True)
        prefix = result.last_hidden_state.shape[1] - grid_h * grid_w
        registers = int(getattr(self.encoder.config, "num_register_tokens", 0))
        cls_count = prefix - registers
        if cls_count < 1 or prefix < registers:
            raise ValueError("DINO output token count does not match patch and register tokens")
        presence = self.presence_head(result.last_hidden_state[:, :cls_count].mean(dim=1))
        # A patch-14 encoder needs a small resample to make the target strides exact.
        base_size = ((ph + 15) // 16, (pw + 15) // 16)
        sizes = [((ph + stride - 1) // stride, (pw + stride - 1) // stride)
                 for stride in (4, 8, 16, 32)]
        features = []
        for level_index, (index, level, size) in enumerate(zip(self.layers, self.pyramid, sizes)):
            tokens = result.hidden_states[index + 1][:, prefix:]
            feature = tokens.transpose(1, 2).reshape(x.shape[0], -1, grid_h, grid_w)
            if feature.shape[-2:] != base_size:
                feature = F.interpolate(feature, size=base_size, mode="bilinear", align_corners=False)
            feature = level(feature, size)
            if self.adapter is not None and str(level_index) in self.adapter.fp_down:
                if feature.shape[-2:] != (grid_h, grid_w):
                    feature = self.adapter.forward_fp(feature, level_index)
            features.append(feature)
        logits = self.decoder(features)
        logits = F.interpolate(logits, size=(ph, pw), mode="bilinear", align_corners=False)
        return {"seg_logits": logits[..., :h, :w], "presence_logits": presence}


def load_teacher_dino(ckpt, device="cpu"):
    """Restore a checkpoint from its saved architecture, without fetching weights."""
    checkpoint = ckpt if isinstance(ckpt, dict) else torch.load(ckpt, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {}).get("model", {})
    model = TeacherDino(backbone=config.get("backbone", "facebook/dinov2-large"),
                        freeze=config.get("freeze", True), unfreeze_last=config.get("unfreeze_last", 0),
                        layers=config.get("layers"), decoder_ch=config.get("decoder_ch", 256),
                        num_labels=config.get("num_labels", 5), input_mode=config.get("input_mode", "rgb"),
                        backbone_config=config.get("backbone_config"), pretrained=False,
                        adapter=config.get("adapter"), adapter_parts=config.get("adapter_parts", "bb+fp"),
                        adapter_dim=config.get("adapter_dim", 64), grad_ckpt=config.get("grad_ckpt", False))
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.source_config = checkpoint.get("config", {})
    return model.to(device).eval()
