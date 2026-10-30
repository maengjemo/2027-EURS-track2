"""Static, single-file ONNX export plus ONNX Runtime and organizer checks."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from eurs.models import ExportModel, build_model


REPO = Path(__file__).resolve().parents[2]
ort.disable_telemetry_events()


def load_model(ckpt, device="cpu", input_blur=0.0):
    """input_blur>0이면 학생 v2 계열에 추론 시 입력 저역통과(set_input_blur)를 건다(재학습 없음, export에 포함)."""
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {})
    name = model_cfg.get("name", "eurs_v0")
    kwargs = {}
    if "width" in model_cfg:
        kwargs["width"] = float(model_cfg["width"])
    # duolite_pre 변형: 백본 구조는 config로 재현하되 가중치는 ckpt에서 오므로
    # timm 다운로드를 막기 위해 pretrained=False로 강제한다.
    if "backbone" in model_cfg:
        kwargs["backbone"] = model_cfg["backbone"]
        kwargs["pretrained"] = False
    if "dec" in model_cfg:
        kwargs["dec"] = model_cfg["dec"]
    if "decoder" in model_cfg:
        kwargs["decoder"] = model_cfg["decoder"]
    for key in ("correct_then_segment", "ctseg_stopgrad", "seg_head", "restore_grad_block_deep",
                "restore_head", "cell_stride", "seg_couple", "couple_control", "couple_detach",
                "dec_widths", "ppm_ch", "ppm_bins", "detail_widths", "ctx_ch",
                "fuse_blocks", "heads", "mlp_ratio", "attn", "iters", "export_safe",
                "distill", "teacher_hidden", "teacher_head_ch", "proj_hidden"):
        if key in model_cfg:
            kwargs[key] = model_cfg[key]
    model = build_model(name, **kwargs)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    if input_blur and float(input_blur) > 0:
        if not hasattr(model, "set_input_blur"):
            raise ValueError(f"{type(model).__name__} has no set_input_blur (student v2 only)")
        model.set_input_blur(float(input_blur))
    return model.to(device).eval()


def apply_calibration(model, calibration_path):
    """calibration.json의 class_bias(R^5)를 모델의 class_bias 파라미터에 굽는다.

    presence_threshold는 export forward에 게이트가 없으므로 굽지 않고 메타데이터로만
    남는다(eval/inference 경로가 참조). class_bias가 없는 모델(eurs_v0)이면 오류를 낸다.
    """
    data = json.loads(Path(calibration_path).read_text())
    bias = data.get("class_bias")
    if bias is None:
        raise ValueError(f"{calibration_path} has no 'class_bias'")
    if not hasattr(model, "class_bias"):
        raise ValueError(f"{type(model).__name__} has no class_bias parameter to bake calibration into")
    with torch.no_grad():
        model.class_bias.copy_(torch.tensor(bias, dtype=model.class_bias.dtype))
    return data


def _static_conv_report(path):
    """Conv/Gemm/MatMul 출력 중 shape inference 후 정적 shape가 없는 노드 수를 센다."""
    model = onnx.shape_inference.infer_shapes(onnx.load(str(path), load_external_data=False))
    shapes = {}
    for vi in list(model.graph.value_info) + list(model.graph.output):
        dims = vi.type.tensor_type.shape.dim
        shapes[vi.name] = bool(dims) and all(d.HasField("dim_value") for d in dims)
    convs = [n for n in model.graph.node if n.op_type in ("Conv", "Gemm", "MatMul")]
    dynamic = sum(1 for n in convs if not shapes.get(n.output[0], False))
    return {"total_convs": len(convs), "dynamic_convs": dynamic}


def export_onnx(model, out, hw=(512, 1024), simplify=True, check=True, strict=False):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if any(side % 128 for side in hw):
        # PPM의 AdaptiveAvgPool(1/2/4)은 1/32 특징 크기가 4의 배수여야 정적 ONNX로 export된다.
        raise ValueError(f"export resolution must be a multiple of 128 on both sides, got {hw}")
    model = model.cpu().eval()
    if hasattr(model, "switch_to_deploy"):
        # RepConv(학생 v2) 학습용 다분기를 단일 3x3으로 합친다. eval 출력은 같다.
        model.switch_to_deploy()
    wrapper = ExportModel(model).eval()
    sample = torch.rand(1, 3, *hw)
    with torch.no_grad():
        reference = wrapper(sample)
        torch.onnx.export(wrapper, sample, str(out), opset_version=17, dynamic_axes=None,
                          input_names=["input"], output_names=["restored_image", "seg_logits"],
                          do_constant_folding=True, external_data=False, dynamo=False)
    graph = onnx.load(str(out), load_external_data=False)
    onnx.checker.check_model(graph)
    if simplify:
        try:
            import onnxsim
        except ImportError as exc:
            raise RuntimeError("--simplify requires onnxsim") from exc
        graph, passed = onnxsim.simplify(graph)
        if not passed:
            raise RuntimeError("onnxsim verification failed")
        onnx.save_model(graph, str(out), save_as_external_data=False)
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input": sample.numpy()})
    argmax_agreement = float((actual[1].argmax(1) == reference[1].numpy().argmax(1)).mean())
    restored_error = float(np.max(np.abs(actual[0] - reference[0].numpy())))
    print(f"ONNX Runtime: seg argmax agreement={argmax_agreement:.6f}; restored max abs error={restored_error:.6g}")
    if argmax_agreement < 0.995 or restored_error > 1e-2:
        raise RuntimeError("ONNX Runtime comparison failed")
    report = {"agreement": argmax_agreement, "max_abs_error": restored_error}
    if check:
        # 규칙: 동적 Resize sizes(Shape→Concat)가 남으면 주최측 검사기의 shape inference가 하류 Conv를
        # 놓쳐 GFLOPs를 과소 집계한다(enet_dec 3.39 vs 실제 11.67, 2026-10-03). 모든 Conv 출력 shape가
        # 정적인지 확인하고, 아니면 onnxsim 단순화 그래프로 다시 재서 큰 값을 기준으로 삼는다.
        report.update(_static_conv_report(out))
        if report["dynamic_convs"]:
            raise SystemExit(f"REJECTED by project rule: {report['dynamic_convs']}/{report['total_convs']} Conv outputs "
                             "have no static shape (organizer checker would under-count FLOPs); export with simplify=True or make sizes static")
        # 규칙(CLAUDE.md "Track 2 제출 게이트"): 주최측 스크립트를 수정 없이 실행하고 PASSED가 아니면 실패.
        # --check_ops 경고(어휘 밖 op)도 TensorRT 컴파일 위험이므로 strict=True에서는 실패로 본다.
        result = subprocess.run([sys.executable, str(REPO / "third_party" / "EURS_check_submission_flops.py"),
                                 "--onnx", str(out), "--check_ops"], check=False, capture_output=True, text=True)
        print(result.stdout, end="")
        (out.with_suffix(".flops.txt")).write_text(result.stdout)  # 제출물 옆에 검사 보고서를 남긴다
        if result.returncode:
            raise SystemExit(result.returncode)
        if strict and "WARNING: op types outside" in result.stdout:
            raise SystemExit("REJECTED by project rule: ops outside organizer vocabulary (see --check_ops warning above)")
        for line in result.stdout.splitlines():
            if line.startswith("Total compute:"):
                report["gflops"] = float(line.split()[2])
            elif line.startswith("Total parameters:"):
                report["params_m"] = float(line.split()[2])
    return report
