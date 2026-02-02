ONNX export guide for this repo

Overview
--------
This repository's `camera_and_boundingboxes.py` can use an ONNX model as a fallback when Ultralytics/PyTorch is not available (useful on ARM devices where PyTorch may cause SIGILL). Use the steps below to export a YOLO model into ONNX on a compatible machine (x86 or an ARM machine with a compatible torch wheel).

Quick steps
-----------
1. Create / activate a Python env on a machine where `ultralytics` and `torch` import successfully.

   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install ultralytics onnx onnxruntime onnxsim

2. Place your `.pt` model in the repo (e.g., `yolov10n.pt`) or reference it by path.

3. Run the helper script from the repo root:

   python scripts/export_yolo_to_onnx.py --model yolov10n.pt --output yolov10n.onnx --opset 12 --simplify

   - `--opset` controls the ONNX opset (12 is a reasonable default).
   - `--simplify` will attempt to run onnxsim simplification if available.

4. Confirm the created `yolov10n.onnx` appears in the repo root.

5. Optional: test loading with onnxruntime:

   python -c "import onnxruntime as ort; sess = ort.InferenceSession('yolov10n.onnx'); print(sess.get_providers())"

Notes & troubleshooting
-----------------------
- If `ultralytics` import or `torch` import fails with `Illegal instruction`, move the export step to a different machine (e.g., your laptop or an x86 cloud instance such as Colab).
- The export script tries the Python API (`YOLO(...).export`) first and falls back to the `ultralytics` module CLI exporter.
- If you need help exporting the model (or want me to run a conversion for you on a compatible environment), tell me where the `.pt` model is and whether you want `opset` / `simplify` options set differently.
