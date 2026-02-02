#!/usr/bin/env python3
"""Decode the base64 sample image and write it to data/sample_image.png
"""
import base64
import os

b64_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_image.b64')
out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_image.png')

b64_path = os.path.normpath(b64_path)
out_path = os.path.normpath(out_path)

os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(b64_path, 'r') as f:
    b64 = f.read().strip()

b = base64.b64decode(b64)
with open(out_path, 'wb') as f:
    f.write(b)

print('Wrote sample image to', out_path)
