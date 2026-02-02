#!/usr/bin/env python3
"""
Simple helper to export a YOLO model (Ultralytics) to ONNX.
Run this on a machine where `ultralytics` and a compatible PyTorch can be imported (x86 or a compatible ARM build).

Examples:
  python scripts/export_yolo_to_onnx.py --model yolov10n.pt --output yolov10n.onnx --opset 12 --simplify

Requirements:
  pip install ultralytics onnx onnxruntime onnxsim

This script tries the Python API first (`YOLO(...).export`) and falls back to the `yolo` CLI if available.
"""

import argparse
import os
import subprocess
import sys


def run_cmd(cmd):
    print("=> Running:", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    return res.returncode


def export_with_ultralytics(model_path, out_path, opset, simplify, imgsz):
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("Could not import ultralytics in Python environment:", e)
        return False

    print("Loaded ultralytics. Creating model object from:", model_path)
    model = YOLO(model_path)
    print("Exporting to ONNX with opset=", opset, " simplify=", simplify)

    # ultralytics' export accepts format='onnx' and several kwargs
    kwargs = {
        "format": "onnx",
        "opset": opset,
        "imgsz": imgsz,
    }
    if simplify:
        kwargs["simplify"] = True

    try:
        # model.export prints progress and returns path(s)
        model.export(**kwargs)
    except Exception as e:
        print("Export via ultralytics API failed:", e)
        return False

    # The ultralytics export places the ONNX next to the .pt by default and/or returns it; try to find file
    if os.path.exists(out_path):
        print("Export completed and found:", out_path)
        return True

    # Try common output name
    out_dir = os.path.dirname(out_path) or "."
    base = os.path.splitext(os.path.basename(model_path))[0]
    candidate = os.path.join(out_dir, base + ".onnx")
    if os.path.exists(candidate):
        if candidate != out_path:
            os.rename(candidate, out_path)
        print("Export completed and found:", out_path)
        return True

    print("Export finished but could not find the ONNX file at the expected location.")
    return False


def export_with_cli(model_path, out_path, opset, simplify, imgsz):
    # Try 'yolo export' CLI
    cmd = [sys.executable, "-m", "ultralytics", "export", "model=" + model_path, "format=onnx", f"opset={opset}", f"imgsz={imgsz}"]
    if simplify:
        cmd.append("simplify=True")
    rc = run_cmd(cmd)
    if rc != 0:
        print("Ultralytics CLI export failed (exit code ", rc, ").")
        return False

    # Move candidate ONNX to requested output if possible
    base = os.path.splitext(os.path.basename(model_path))[0]
    candidate = base + ".onnx"
    if os.path.exists(candidate):
        if candidate != out_path:
            os.replace(candidate, out_path)
        print("Export completed and found:", out_path)
        return True

    print("Export via CLI completed but could not find", candidate)
    return False


def sanity_check_onnx(path):
    try:
        import onnx
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        print("ONNX file passes basic onnx.checker checks.")
    except Exception as e:
        print("ONNX sanity check failed:", e)
        return False

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(path)
        print("onnxruntime can load the model: OK. Providers:", sess.get_providers())
    except Exception as e:
        print("onnxruntime failed to load the model:", e)
        return False

    return True


def main():
    p = argparse.ArgumentParser(description="Export YOLO model to ONNX")
    p.add_argument("--model", required=True, help="Path to source model (.pt) or a model name")
    p.add_argument("--output", default=None, help="Desired ONNX output path (defaults to <model_base>.onnx)")
    p.add_argument("--opset", type=int, default=12, help="ONNX opset to use (default: 12)")
    p.add_argument("--simplify", action="store_true", help="Try to simplify the ONNX graph with onnx-simplifier")
    p.add_argument("--imgsz", type=int, default=640, help="Input image size used for export (default 640)")
    args = p.parse_args()

    model_path = args.model
    out_path = args.output or (os.path.splitext(os.path.basename(model_path))[0] + ".onnx")

    # Try python API first
    ok = export_with_ultralytics(model_path, out_path, args.opset, args.simplify, args.imgsz)
    if not ok:
        print("Python API export failed or unavailable; trying CLI method...")
        ok = export_with_cli(model_path, out_path, args.opset, args.simplify, args.imgsz)

    if not ok:
        print("Export failed. Ensure you run this on a machine with a compatible PyTorch/ultralytics install.")
        sys.exit(2)

    print("Running basic ONNX checks...")
    if not sanity_check_onnx(out_path):
        print("Warning: ONNX file failed basic checks. It may still work in some runtimes.")
    else:
        print("ONNX export successful and validated:", out_path)


if __name__ == '__main__':
    main()
