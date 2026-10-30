"""Track 1 DINOv3 ConvNeXt + UperNet segmentation teacher."""

import torch
from torch import nn
from torch.nn import functional as F

from eurs.models.teacher_dino import _UperDecoder


class TeacherConvNeXt(nn.Module):
    def __init__(self, backbone="facebook/dinov3-convnext-large-pretrain-lvd1689m",
                 pretrained=True, backbone_config=None, decoder_ch=512, num_labels=5,
                 drop_path=0.2):
        super().__init__()
        from transformers import DINOv3ConvNextConfig, DINOv3ConvNextModel

        if backbone_config is not None:
            config = DINOv3ConvNextConfig(**backbone_config)
            self.encoder = DINOv3ConvNextModel(config)
        elif pretrained:
            self.encoder = DINOv3ConvNextModel.from_pretrained(
                backbone, drop_path_rate=float(drop_path))
        else:
            config = DINOv3ConvNextConfig(drop_path_rate=float(drop_path))
            self.encoder = DINOv3ConvNextModel(config)
        config = self.encoder.config
        if config.model_type != "dinov3_convnext" or len(config.hidden_sizes) != 4:
            raise ValueError("TeacherConvNeXt requires four DINOv3 ConvNeXt stages")
        self.backbone = backbone
        self.input_mode = "rgb"
        self.decoder_ch = int(decoder_ch)
        self.num_labels = int(num_labels)
        self.drop_path = float(config.drop_path_rate)
        self.projections = nn.ModuleList(
            nn.Conv2d(ch, self.decoder_ch, 1) for ch in config.hidden_sizes)
        self.decoder = _UperDecoder(self.decoder_ch, self.num_labels)
        self.presence_head = nn.Linear(config.hidden_sizes[-1], self.num_labels)
        self.register_buffer("image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
                             persistent=False)

    def config_dict(self):
        return {"name": "teacher_cnx", "backbone": self.backbone,
                "backbone_config": self.encoder.config.to_dict(),
                "decoder_ch": self.decoder_ch, "num_labels": self.num_labels,
                "drop_path": self.drop_path, "input_mode": self.input_mode}

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("TeacherConvNeXt expects (B, 3, H, W) input")
        h, w = x.shape[-2:]
        padded = F.pad(x, (0, (-w) % 32, 0, (-h) % 32), mode="replicate")
        normalized = (padded - self.image_mean) / self.image_std
        output = self.encoder(normalized, output_hidden_states=True, return_dict=True)
        stages = output.hidden_states[1:]
        if len(stages) != 4:
            raise ValueError("DINOv3 ConvNeXt did not return four stage features")
        features = [project(stage) for project, stage in zip(self.projections, stages)]
        logits = self.decoder(features)
        logits = F.interpolate(logits, size=padded.shape[-2:], mode="bilinear", align_corners=False)
        presence = self.presence_head(stages[-1].mean(dim=(-2, -1)))
        return {"seg_logits": logits[..., :h, :w], "presence_logits": presence}


def load_teacher_cnx(ckpt, device="cpu"):
    """Restore saved architecture and weights without fetching the backbone."""
    checkpoint = ckpt if isinstance(ckpt, dict) else torch.load(
        ckpt, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {}).get("model", {})
    if config.get("backbone_config") is None:
        raise ValueError("TeacherConvNeXt checkpoint requires backbone_config")
    model = TeacherConvNeXt(backbone=config.get("backbone", "facebook/dinov3-convnext-large-pretrain-lvd1689m"),
                           pretrained=False, backbone_config=config["backbone_config"],
                           decoder_ch=config.get("decoder_ch", 512),
                           num_labels=config.get("num_labels", 5),
                           drop_path=config.get("drop_path", 0.2))
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.source_config = checkpoint.get("config", {})
    return model.to(device).eval()
