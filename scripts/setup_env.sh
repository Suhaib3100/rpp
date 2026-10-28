#!/usr/bin/env bash
# Create the SAGE-HE virtual environment.
#
# Torch is pulled from the CPU wheel index: this project was developed and
# trained on a 4-vCPU machine with no CUDA device. On a CUDA host, drop the
# --index-url line and plain `pip install -r requirements.txt` will fetch the
# GPU build instead; nothing else in the project needs to change (device
# selection is automatic, see src/utils/device.py).
set -euo pipefail

PYTHON=${PYTHON:-/usr/bin/python3.11}
VENV=${VENV:-.venv}

"$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip wheel setuptools

# CPU-only torch stack first, so the generic index cannot shadow it with a
# CUDA build that would pull ~2.5 GB of unusable nvidia wheels.
"$VENV/bin/pip" install \
  torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cpu

"$VENV/bin/pip" install -r requirements.txt

echo
echo "Environment ready: $VENV"
"$VENV/bin/python" -c "
import torch, tenseal, timm, cv2, sklearn
print('torch   ', torch.__version__, '| cuda:', torch.cuda.is_available())
print('tenseal ', tenseal.__version__)
print('timm    ', timm.__version__)
print('opencv  ', cv2.__version__)
print('sklearn ', sklearn.__version__)
"
