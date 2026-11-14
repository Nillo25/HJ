#!/usr/bin/env bash
# CubiCasa5K model for stoa/ (masks for plan_check votes and big-sheet auto scale). Run from the repo root.
set -euo pipefail
mkdir -p third_party runs/cubicasa
.venv/bin/pip install -q torch torchvision lmdb svgpathtools gdown scikit-image
[ -d third_party/CubiCasa5k ] || git clone -q --depth 1 https://github.com/CubiCasa/CubiCasa5k.git third_party/CubiCasa5k
# scipy>=1.11: stats.mode(...).mode is a scalar
sed -i.bak 's/stats\.mode(\([a-z_]*\))\.mode\[0\]/np.atleast_1d(stats.mode(\1).mode)[0]/g' third_party/CubiCasa5k/floortrans/post_prosessing.py
# weights (Google Drive, link from the CubiCasa5k README)
[ -f runs/cubicasa/model_best_val_loss_var.pkl ] || .venv/bin/python -m gdown 1gRB7ez1e4H7a9Y09lLqRuna0luZO5VRK -O runs/cubicasa/model_best_val_loss_var.pkl
echo "CubiCasa ready"
