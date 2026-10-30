"""Track 1 DINOv3 ViT teacher with attention LoRA and a last-layer head."""

import math

import torch
from torch import nn
from torch.nn import functional as F



class LoRALinear(nn.Module):
    """Frozen linear projection plus a zero-initialized low-rank update."""

    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.0):
        super().__init__()
        if not isinstance(base, nn.Linear) or r < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("LoRA requires a linear base, r >= 1, alpha > 0, and 0 <= dropout < 1")
        self.base = base.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scale


def replace_attention_projections(encoder, r=16, alpha=32, targets=("q", "v")):
    """Replace the named HF DINOv3 ViT attention projections in every block."""
    names = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj"}
    targets = tuple(targets)
    if not targets or len(set(targets)) != len(targets) or set(targets) - names.keys():
        raise ValueError("lora_targets must be unique members of q, k, v, o")
    if encoder.config.model_type != "dinov3_vit":
        raise ValueError("TeacherDinoLoRA requires a DINOv3 ViT backbone")
    blocks = encoder.model.layer
    if not blocks:
        raise ValueError("DINOv3 ViT has no transformer blocks")
    for block in blocks:
        attention = block.attention
        for target in targets:
            name = names[target]
            base = getattr(attention, name, None)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"DINOv3 attention {name} must be nn.Linear")
            setattr(attention, name, LoRALinear(base, r=r, alpha=alpha))
    return encoder


class _DPTLastLayerHead(nn.Module):
    """Single-level DPT-style projection and stride-16 to stride-4 upsampling."""

    def __init__(self, hidden, channels, num_labels):
        super().__init__()
        self.project = nn.Conv2d(hidden, channels, 1)
        self.up1 = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                 nn.BatchNorm2d(channels), nn.GELU())
        self.up2 = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                 nn.BatchNorm2d(channels), nn.GELU())
        self.classifier = nn.Conv2d(channels, num_labels, 1)

    def forward(self, x, film=None):
        x = self.project(x)
        if film is not None:
            x = film(x)
        x = self.up1(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        x = self.up2(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        return self.classifier(x)


class TeacherDinoLoRA(nn.Module):
    """Frozen DINOv3 ViT with LoRA and a last-layer-only segmentation head.

    For ViT-L (24 blocks, hidden size 1024), r=16 on q and v adds exactly
    24 * 2 * 16 * (1024 + 1024) = 1,572,864 LoRA parameters. The DPT
    head with 256 channels adds 1,443,845 more, plus 5,125 for presence:
    3,021,834 trainable parameters in total.
    These are analytic counts; pretrained ViT-L weights are never fetched for
    this calculation.
    """

    def __init__(self, backbone="facebook/dinov3-vitl16-pretrain-lvd1689m", pretrained=True,
                 backbone_config=None, lora_r=16, lora_alpha=32, lora_targets=("q", "v"),
                 head="linear", head_ch=256, num_labels=5, grad_ckpt=True,
                 adapter="lora", adapter_dim=64, meta_cond=False, input_mode="rgb"):
        super().__init__()
        if input_mode not in ("rgb", "restored", "concat"):
            raise ValueError("input_mode must be rgb, restored or concat")
        from transformers import AutoConfig, AutoModel, DINOv3ViTConfig
        if backbone_config is not None:
            config_data = dict(backbone_config)
            model_type = config_data.pop("model_type")
            config = AutoConfig.for_model(model_type, **config_data)
            self.encoder = AutoModel.from_config(config)
        elif pretrained:
            self.encoder = AutoModel.from_pretrained(backbone)
        else:
            self.encoder = AutoModel.from_config(DINOv3ViTConfig())
        if self.encoder.config.model_type != "dinov3_vit":
            raise ValueError("TeacherDinoLoRA requires a DINOv3 ViT backbone")
        if head not in ("linear", "dpt") or head_ch < 1:
            raise ValueError("head must be linear or dpt, and head_ch must be positive")
        if adapter not in ("lora", "deba", "none"):
            raise ValueError("adapter must be 'lora', 'deba', or 'none'")
        self.backbone = backbone
        self.input_mode = input_mode
        self.patch_size = int(self.encoder.config.patch_size)
        self.num_labels = int(num_labels)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.lora_targets = tuple(lora_targets)
        self.head_name = head
        self.head_ch = int(head_ch)
        self.grad_ckpt = bool(grad_ckpt)
        self.adapter_name = adapter
        self.adapter_dim = int(adapter_dim)
        self.meta_cond = bool(meta_cond)
        self.encoder.requires_grad_(False)
        if adapter == "lora":
            replace_attention_projections(self.encoder, self.lora_r, self.lora_alpha, self.lora_targets)
        hidden = int(self.encoder.config.hidden_size)
        if adapter == "deba":
            from .deba import DeBA
            self.adapter = DeBA(hidden, len(self.encoder.model.layer), 0, (), d_b=self.adapter_dim)
        else:
            self.adapter = None
        self._adapter_grid = None
        if self.adapter is not None:
            for index, block in enumerate(self.encoder.model.layer):
                block.register_forward_hook(self._bb_hook(index))
        # concat 모드: 6채널(열화 3 + 복원 3)을 동결 패치 임베딩에 넣기 전에 학습 가능한
        # 1x1 conv로 3채널로 투영한다. 열화 채널은 항등, 복원 채널은 0으로 초기화하여
        # 학습 시작 시점 출력이 rgb 모드와 동일하다. 이 투영은 lr_decoder 그룹에 속한다.
        self.input_proj = None
        if input_mode == "concat":
            self.input_proj = nn.Conv2d(6, 3, 1)
            nn.init.zeros_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)
            with torch.no_grad():
                for channel in range(3):
                    self.input_proj.weight[channel, channel, 0, 0] = 1.0
        self.feature_project = nn.Conv2d(hidden, head_ch, 1) if self.meta_cond and head == "linear" else None
        self.film = nn.Linear(11, 2 * head_ch) if self.meta_cond else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        self.head = (nn.Conv2d(head_ch if self.meta_cond else hidden, num_labels, 1) if head == "linear"
                     else _DPTLastLayerHead(hidden, head_ch, num_labels))
        self.presence_head = nn.Linear(hidden, num_labels)
        if self.grad_ckpt:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.register_buffer("image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
                             persistent=False)
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self._set_checkpoint_block_mode(mode)
        return self

    def _set_checkpoint_block_mode(self, mode):
        if self.grad_ckpt and mode:
            # HF checkpoints only blocks in training mode. Keep their frozen
            # children (and LoRA dropout) in eval mode, as in TeacherDino.
            for block in self.encoder.model.layer:
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

    def lora_parameters(self):
        for module in self.encoder.modules():
            if isinstance(module, LoRALinear):
                yield module.lora_A
                yield module.lora_B

    def adapter_parameters(self):
        if self.adapter_name == "lora":
            yield from self.lora_parameters()
        elif self.adapter is not None:
            yield from self.adapter.parameters()

    def _apply_film(self, feature, meta):
        gamma, beta = self.film(meta.to(device=feature.device, dtype=feature.dtype)).chunk(2, dim=1)
        return feature * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def trainable_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def config_dict(self):
        return {"name": "teacher_dino_lora", "backbone": self.backbone,
                "backbone_config": self.encoder.config.to_dict(), "input_mode": self.input_mode,
                "lora_r": self.lora_r, "lora_alpha": self.lora_alpha,
                "lora_targets": list(self.lora_targets), "head": self.head_name,
                "head_ch": self.head_ch, "num_labels": self.num_labels,
                "grad_ckpt": self.grad_ckpt, "adapter": self.adapter_name,
                "adapter_dim": self.adapter_dim, "meta_cond": self.meta_cond}

    def forward(self, x, meta=None):
        expected = 6 if self.input_mode == "concat" else 3
        if x.ndim != 4 or x.shape[1] != expected:
            raise ValueError(f"TeacherDinoLoRA {self.input_mode} expects (B, {expected}, H, W) input")
        if self.meta_cond and (meta is None or meta.shape != (x.shape[0], 11)):
            raise ValueError("meta_cond=True requires a (B,11) metadata tensor")
        if self.input_mode == "concat":
            x = self.input_proj(x)
        h, w = x.shape[-2:]
        p = self.patch_size
        padded = F.pad(x, (0, (-w) % p, 0, (-h) % p), mode="replicate")
        ph, pw = padded.shape[-2:]
        self._adapter_grid = (ph // p, pw // p)
        result = self.encoder((padded - self.image_mean) / self.image_std, return_dict=True)
        grid_h, grid_w = ph // p, pw // p
        tokens = result.last_hidden_state
        prefix = tokens.shape[1] - grid_h * grid_w
        registers = int(getattr(self.encoder.config, "num_register_tokens", 0))
        cls_count = prefix - registers
        if cls_count < 1 or prefix < registers:
            raise ValueError("DINO output token count does not match patch and register tokens")
        presence = self.presence_head(tokens[:, :cls_count].mean(dim=1))
        patches = tokens[:, prefix:].transpose(1, 2).reshape(x.shape[0], -1, grid_h, grid_w)
        if self.head_name == "linear":
            if self.meta_cond:
                patches = self._apply_film(self.feature_project(patches), meta)
            logits = self.head(patches)
        else:
            logits = self.head(patches, (lambda feature: self._apply_film(feature, meta))
                               if self.meta_cond else None)
        logits = F.interpolate(logits, size=(ph, pw), mode="bilinear", align_corners=False)
        return {"seg_logits": logits[..., :h, :w], "presence_logits": presence}


def load_teacher_dino_lora(ckpt, device="cpu"):
    """Restore architecture and all saved weights without downloading a backbone."""
    checkpoint = ckpt if isinstance(ckpt, dict) else torch.load(ckpt, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {}).get("model", {})
    if config.get("name") != "teacher_dino_lora" or not config.get("backbone_config"):
        raise ValueError("TeacherDinoLoRA checkpoint requires model.name and backbone_config")
    model = TeacherDinoLoRA(backbone=config.get("backbone", "facebook/dinov3-vitl16-pretrain-lvd1689m"),
                            pretrained=False, backbone_config=config["backbone_config"],
                            lora_r=config.get("lora_r", 16), lora_alpha=config.get("lora_alpha", 32),
                            lora_targets=config.get("lora_targets", ("q", "v")),
                            head=config.get("head", "linear"), head_ch=config.get("head_ch", 256),
                            num_labels=config.get("num_labels", 5), grad_ckpt=config.get("grad_ckpt", True),
                            adapter=config.get("adapter", "lora"), adapter_dim=config.get("adapter_dim", 64),
                            meta_cond=config.get("meta_cond", False),
                            input_mode=config.get("input_mode", "rgb"))
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.source_config = checkpoint.get("config", {})
    return model.to(device).eval()
