# EURS @ MaCVi WACV 2027: Track 2 reproduction (team GIST, submission #19347)

```bash
python repro_check.py . --export-onnx      # regenerates the ONNX from weights/last.pth and compares it with onnx/submitted_model.onnx
```
- `export_onnx.py`: `main(model_dir, out_onnx)`; `src/eurs/`: model code; `weights/last.pth`: checkpoint (EMA weights + config);
  `onnx/submitted_model.onnx`: the leaderboard file. Method and data: `METHOD.md`.
- Built from the organizers' template <https://github.com/CeviKle/EURS-MaCVi2027-repro-template>.
