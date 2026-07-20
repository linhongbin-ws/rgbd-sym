#!/bin/bash
# One-shot installer: the ORIGINAL equi-rl-for-pomdps baseline into the
# `equi-pomdp` conda env (torch 1.12.0 already installed there).
#
# Mirrors the PROVEN-working rgbd-sym env on this machine rather than the
# readme verbatim:
#   - mujoco-py / lie_learn / torch-geometric SKIPPED: block domains are
#     pybullet-based and the working env runs without them
#   - numpy pinned 1.23.0: env ships numpy 2.0.1, torch 1.12 breaks on numpy 2
#   - gym==0.21.0 only builds under old pip/setuptools/wheel -> pinned first
#
# Must run OUTSIDE the sandbox (conda-env write + network):
#     bash bash/setup_equipomdp_original.sh
set -e
source "$HOME/miniconda3/bin/activate"
conda activate equi-pomdp
cd "$(dirname "$0")/../ext/equi-rl-for-pomdps-original"

# gym 0.21 cannot build under modern packaging tools
pip install "setuptools==65.5.0" "wheel==0.38.4" "pip==23.3.2"

# torch 1.12 is numpy-1.x only
pip install numpy==1.23.0

# top-level requirements.txt minus mujoco-py; versions matched to rgbd-sym env
pip install scikit-learn "future-fstrings==1.2.0" "gym==0.21.0" \
    "ruamel-yaml==0.16.12" "absl-py==0.11.0" matplotlib seaborn \
    more_itertools tensorboardX PyYAML wandb psutil
pip install Box2D || echo "WARN: Box2D failed -- optional, block tasks don't use it"

# escnn runtime deps (lie_learn/torch-geometric intentionally omitted)
pip install joblib pymanopt autograd scipy
pip install -e ./escnn --no-deps

# pybullet block domains
pip install attrdict GitPython "pybullet==3.0.9" scikit-image tqdm
pip install -e ./pomdp_robot_domains --no-deps
pip install -e ./pomdp-domains --no-deps

# smoke test: imports + env registration
python - <<'EOF'
import numpy, torch, gym, escnn, pybullet
print("numpy", numpy.__version__, "| torch", torch.__version__,
      "| gym", gym.__version__, "| pybullet OK | escnn", escnn.__version__)
import helping_hands_rl_envs, pdomains
print("helping_hands_rl_envs <-", helping_hands_rl_envs.__file__)
print("pdomains              <-", pdomains.__file__)
EOF
echo "=== setup done. Train with: source bash/init_equipomdp_original.sh ==="
