# onnx/ (Track 2 only; delete this folder if you entered only Track 1)

Put a copy of the exact ONNX file you uploaded to the leaderboard for your reported submission here, named `submitted_model.onnx`:

    onnx/submitted_model.onnx

Then run `python repro_check.py . --export-onnx`: it regenerates the file with your `export_onnx.py` and reports whether it equals this copy.
Write the file's SHA-256 (`sha256sum onnx/submitted_model.onnx`) into your factsheet. The model has at most 1.0 M parameters, so the file is
far below GitHub's 100 MB limit for a normal file.
