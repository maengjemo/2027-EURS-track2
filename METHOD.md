# Method description

Team: GIST   Track(s): 2 (this repository; Track 1: https://github.com/maengjemo/2027-EURS-track1)   Reported testing-phase submission: #19347 (composite 0.383)

## Approach
A single lightweight joint network (0.881 M parameters, 4.15 GFLOPs at 640x640) predicts the restored image and 5-class segmentation logits from one degraded RGB image.

- **Encoder**: timm `lcnet_075` (ImageNet-1k pretrained `lcnet_075.ra2_in1k`, strides 4/8/16/32). Hardswish / Hardsigmoid / SE pooling are
  rewritten with ONNX ops in the organizer vocabulary (Mul, Clip, Add, MatMul, Reshape; no Div), values unchanged.
- **Global context**: a global-context block on the stride-32 features and a pooled context vector, which feeds (a) FiLM modulation of the
  decoder at strides 16 and 8, (b) a class-presence head that adds a gated bias to the segmentation logits, (c) a global colour-correction
  head, and (d) a water-type / depth head used only as an auxiliary training loss.
- **Decoder (DS-FPN)**: pyramid pooling module (bins 1/2/3/6) on stride 32, then top-down fusion with lateral 1x1 convolutions and two
  depthwise-separable blocks per level at strides 16/8/4/2 (96/64/48/32 channels); the stride-2 level also takes a separate stem convolution of the image.
- **Segmentation**: 1x1 head at stride 2 (+ learned class bias + presence gate), bilinear upsampling to the input size.
- **Restoration**: (1) global colour correction of the input with per-image parameters from the context vector (channel gains and offsets,
  a 3x3 colour matrix and a piecewise-linear tone curve), (2) plus a residual from the stride-2 decoder features (1x1 conv + PixelShuffle x2),
  (3) plus a full-resolution refiner (re-parameterised 3x3 convolutions) on [input, image]; output clipped to [0, 1].
- **Training-only parts** (not in the ONNX): online knowledge distillation from a frozen segmentation teacher (pixel-wise KL, T = 4, on the
  external images only). The teacher is our Track 1 combined teacher: frozen DINOv3 ViT-L/16 + LoRA (q, v, r = 16) + DeBA adapter + DPT head
  with water-type / depth FiLM, trained on 16 EURS scenes plus the 5 external datasets below. The checkpoint also contains feature-projection
  heads (`distill_heads`) of the model class; they were not used for this model (no encoder pre-distillation) and are not exported.
- Code: `src/eurs/models/student_distill.py`, `src/eurs/models/student_v2.py` (`StudentDSFPN`), `src/eurs/models/eurs_v0.py` (`ExportModel`).

## Training data
- **EURS training set, all 20 scenes** (no held-out scene for this final model). Model selection was done earlier on a 16/4 scene split
  (validation scenes fish12, fish20, sharkFish, turtles1) with the same recipe; the final weights are the EMA weights after the last epoch (60).
- **External data (disclosed with "Other" on the leaderboard)**, remapped to the 5 EURS classes; pixels without a usable label were filled
  with pseudo labels of the frozen teacher (confidence-thresholded) and every external image was re-degraded with a fitted
  underwater image-formation model (RSUIGM parameters fitted on EURS):
  Coralscapes, UIIS10K, USIS10K, SUIM, USIS16K.
- **Pretrained models**: timm `lcnet_075.ra2_in1k` (ImageNet-1k, Apache-2.0); teacher backbone `facebook/dinov3-vitl16-pretrain-lvd1689m`
  (DINOv3 license; used only to train the teacher and as distillation target, not part of the submitted model).
- No testing-phase image was used for training, pseudo-labelling or model selection.
- Recipe: 60 epochs, batch 8, crops 512x1024 with scale 0.75-1.25, copy-paste of elasmobranch / other / fish objects (p = 0.5),
  AdamW lr 2e-3 (encoder 2e-4), weight decay 0.01, 1 warm-up epoch, EMA 0.999. Losses: CE + Lovasz with class weights (1, 1, 2, 4, 3),
  Charbonnier + SSIM restoration loss (weight 1.0), water-type / depth loss (0.1), class-presence loss (0.2), online KD (1.0); losses of the external samples weighted 0.5. fp32. Full config is stored inside `weights/last.pth` (`config`). Seed 0 (trainer default).

## Anything non-standard at inference
None. Fixed input 640x640 (the leaderboard harness resizes the image), no TTA, no ensembling, no post-processing in or after the graph.

## Reproduction notes
- `python export_onnx.py` / `main(model_dir, out_onnx)` rebuilds the network from the config stored in the checkpoint, loads the EMA weights,
  exports with `torch.onnx.export` (opset 17, static input (1, 3, 640, 640), constant folding) and simplifies with `onnxsim` (so that all
  Resize / Conv shapes are static). CPU only, about 20 s.
- `onnx/submitted_model.onnx` is the file uploaded for #19347 (SHA-256 `ba702418259395ec8244de289479207f2be8fec3475a7b8ae9f79a32e4cdf3ed`).
  Organizer checker: 0.8811 M parameters, 4.1506 GFLOPs, PASSED.
- The original file was exported with PyTorch 2.11 / onnxsim 0.8.1. Re-exporting gives identical ONNX Runtime outputs (max abs difference 0).
- Training code (not needed for reproduction): private development repository, available on request.
