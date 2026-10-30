"""Track 2: regenerate the submitted ONNX (testing-phase submission #19347) from weights/last.pth.

The organizers call   main(model_dir, out_onnx).
Steps (identical to the leaderboard file's export, src/eurs/export/onnx_export.py):
  1. rebuild the student from the config stored in the checkpoint (timm backbone with pretrained=False: no download),
  2. load the EMA weights, switch RepConv branches to deploy form if present, wrap with ExportModel
     (outputs: restored_image in [0, 1], seg_logits),
  3. torch.onnx.export, opset 17, static input (1, 3, 640, 640), constant folding,
  4. onnxsim.simplify (makes every Resize/Conv shape static so the organizer FLOPs checker counts all Convs).
"""
import inspect
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from eurs.models import ExportModel, build_model  # noqa: E402

HW = (640, 640)
CKPT = "last.pth"
MODEL_KEYS = ("dec_widths", "ppm_ch", "ppm_bins", "detail_widths", "ctx_ch", "fuse_blocks", "heads", "mlp_ratio",
              "attn", "iters", "export_safe", "distill", "teacher_hidden", "teacher_head_ch", "proj_hidden")


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_cfg = checkpoint.get("config", {}).get("model", {})
    kwargs = {}
    if "width" in model_cfg:
        kwargs["width"] = float(model_cfg["width"])
    if "backbone" in model_cfg:
        kwargs["backbone"] = model_cfg["backbone"]
        kwargs["pretrained"] = False  # weights come from the checkpoint; never download
    for key in MODEL_KEYS:
        if key in model_cfg:
            kwargs[key] = model_cfg[key]
    model = build_model(model_cfg.get("name", "student_distill"), **kwargs)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    return model.cpu().eval()


def main(model_dir, out_onnx):
    import onnx
    import onnxsim

    torch.manual_seed(0)
    model = load_model(os.path.join(model_dir, CKPT))
    if hasattr(model, "switch_to_deploy"):
        model.switch_to_deploy()
    wrapper = ExportModel(model).eval()
    sample = torch.rand(1, 3, *HW)
    os.makedirs(os.path.dirname(os.path.abspath(out_onnx)), exist_ok=True)
    extra = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    with torch.no_grad():
        torch.onnx.export(wrapper, sample, out_onnx, opset_version=17, dynamic_axes=None,
                          input_names=["input"], output_names=["restored_image", "seg_logits"],
                          do_constant_folding=True, **extra)
    graph = onnx.load(out_onnx)
    onnx.checker.check_model(graph)
    graph, passed = onnxsim.simplify(graph)
    if not passed:
        raise RuntimeError("onnxsim verification failed")
    onnx.save_model(graph, out_onnx)
    return out_onnx


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "weights", sys.argv[2] if len(sys.argv) > 2 else "exported.onnx")
