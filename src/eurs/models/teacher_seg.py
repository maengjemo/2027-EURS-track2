"""Track 1 분할 교사: HuggingFace SegFormer(MiT 인코더 + all-MLP 디코더).

Track 1은 단일 모델·연산 제약이 없으므로 분할 교사와 복원 교사를 분리한다. 이 모듈은
ImageNet-1k 사전학습 MiT 인코더(기본 ``nvidia/mit-b5``)를 쓰며, 사전학습 가중치는 HF
캐시(``~/.cache/huggingface``)로 내려받는다. 외부 사전학습 사용은 제출 시 공시 대상이다.

``transformers``는 무거우므로 모듈 최상위에서 import하지 않고 생성자에서 지연 import한다.
따라서 이 파일을 import하는 것만으로 transformers를 요구하지 않는다.
"""
import torch
from torch import nn
from torch.nn import functional as F


class TeacherSeg(nn.Module):
    """SegformerForSemanticSegmentation(num_labels=5) + presence 보조 헤드.

    presence_logits는 마지막 인코더 스테이지 특징의 GAP에 선형층을 붙여 얻는다(BCE 대상).
    매질 헤드는 두지 않는다.
    """

    def __init__(self, backbone="nvidia/mit-b5", num_labels=5, pretrained=True,
                 seg_config=None, input_mode="rgb"):
        super().__init__()
        if input_mode not in ("rgb", "restored", "concat"):
            raise ValueError("input_mode must be rgb, restored or concat")
        self.input_mode = input_mode
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        if seg_config is not None:
            # 저장된 config로 아키텍처만 복원(사전학습 다운로드 없이). load_state_dict가 뒤따른다.
            config = SegformerConfig(**seg_config)
            self.seg = SegformerForSemanticSegmentation(config)
        elif pretrained:
            self.seg = SegformerForSemanticSegmentation.from_pretrained(
                backbone, num_labels=num_labels, ignore_mismatched_sizes=True)
        else:
            # 네트워크 없이 기본(mit-b0 규모) config로 구성. 테스트·오프라인용.
            config = SegformerConfig(num_labels=num_labels)
            self.seg = SegformerForSemanticSegmentation(config)
        if input_mode == "concat" and self.seg.config.num_channels == 3:
            backbone_model = self.seg.segformer
            patch = (backbone_model.stages[0].patch_embeddings if hasattr(backbone_model, "stages")
                     else backbone_model.encoder.patch_embeddings[0])
            old = patch.proj
            expanded = nn.Conv2d(6, old.out_channels, old.kernel_size, old.stride,
                                 old.padding, bias=old.bias is not None)
            with torch.no_grad():
                expanded.weight[:, :3].copy_(old.weight * 0.5)
                expanded.weight[:, 3:].copy_(old.weight * 0.5)
                if old.bias is not None:
                    expanded.bias.copy_(old.bias)
            patch.proj = expanded
            self.seg.config.num_channels = 6
        self.backbone = backbone
        self.num_labels = num_labels
        hidden = self.seg.config.hidden_sizes[-1]
        self.presence_head = nn.Linear(hidden, num_labels)

    def config_dict(self):
        """체크포인트에 저장할 모델 config. 로드 시 사전학습 다운로드 없이 복원한다."""
        return {"name": "teacher_seg", "backbone": self.backbone,
                "num_labels": self.num_labels, "input_mode": self.input_mode,
                "seg_config": self.seg.config.to_dict()}

    def forward(self, x, return_encoder_features=False):
        expected = 6 if self.input_mode == "concat" else 3
        if x.shape[1] != expected:
            raise ValueError(f"{self.input_mode} expects {expected} input channels")
        h, w = x.shape[-2:]
        encoder = self.seg.segformer(x, output_hidden_states=True, return_dict=True)
        features = encoder.hidden_states
        logits = self.seg.decode_head(features)
        seg_logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        pooled = features[-1].mean(dim=(2, 3))
        presence_logits = self.presence_head(pooled)
        output = {"seg_logits": seg_logits, "presence_logits": presence_logits}
        if return_encoder_features:
            output["encoder_features"] = features
        return output


def load_teacher_seg(ckpt, device="cpu"):
    """체크포인트에서 분할 교사를 복원한다(사전학습 재다운로드 없음)."""
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    model_cfg = checkpoint.get("config", {}).get("model", {})
    if model_cfg.get("name") == "teacher_dino":
        from .teacher_dino import load_teacher_dino
        return load_teacher_dino(checkpoint, device)
    if model_cfg.get("name") == "teacher_dino_lora":
        from .teacher_dino_lora import load_teacher_dino_lora
        return load_teacher_dino_lora(checkpoint, device)
    if model_cfg.get("name") == "teacher_cnx":
        from .teacher_cnx import load_teacher_cnx
        return load_teacher_cnx(checkpoint, device)
    seg_config = model_cfg.get("seg_config")
    if seg_config is not None:
        model = TeacherSeg(seg_config=seg_config, input_mode=model_cfg.get("input_mode", "rgb"))
    else:
        model = TeacherSeg(backbone=model_cfg.get("backbone", "nvidia/mit-b5"), pretrained=False,
                           input_mode=model_cfg.get("input_mode", "rgb"))
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.source_config = checkpoint.get("config", {})
    return model.to(device).eval()


def combine_seg_input(degraded, restored, input_mode):
    if input_mode == "rgb":
        return degraded
    if restored is None:
        raise ValueError(f"{input_mode} requires a restored image")
    if restored.shape != degraded.shape:
        raise ValueError("restored and degraded shapes differ")
    return restored if input_mode == "restored" else torch.cat((degraded, restored), dim=1)


class FrozenRestorer(nn.Module):
    """캐스케이드 분할의 고정 복원기. 전체 영상 추론 후 동일 크롭을 추출한다."""

    def __init__(self, ckpt, device="cpu"):
        super().__init__()
        from .teacher_restore import load_teacher_restore
        self.model = load_teacher_restore(ckpt, device)
        self.model.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def forward(self, degraded, full_images=None, crop_boxes=None, water=None, depth=None):
        from .teacher_restore import metadata_vector
        if self.model.use_meta:
            if water is None or depth is None:
                raise ValueError("metadata restore checkpoint requires water and depth")
            meta = metadata_vector(water, depth, device=degraded.device)
        else:
            meta = None
        def restore_whole(image, one_meta):
            h, w = image.shape[-2:]
            padded = F.pad(image, (0, (-w) % 32, 0, (-h) % 32), mode="replicate")
            return self.model(padded, meta=one_meta)["restored_image"][..., :h, :w]
        if full_images is None:
            return restore_whole(degraded, meta).detach()
        if crop_boxes is None or len(full_images) != len(crop_boxes):
            raise ValueError("full_images requires matching crop_boxes")
        patches = []
        for i, full in enumerate(full_images):
            full = full.to(degraded.device)[None]
            restored = restore_whole(full, meta[i:i + 1] if meta is not None else None)
            x0, y0, x1, y1 = [int(v) for v in crop_boxes[i]]
            patches.append(restored[..., y0:y1, x0:x1])
        return torch.cat(patches, dim=0).detach()
