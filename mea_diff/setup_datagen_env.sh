#!/usr/bin/env bash
# Create the SEPARATE MimicGen datagen conda env for MEA v2 (diffusion route).
#
# WHY SEPARATE: EquiDiff pins a customized OLD robomimic fork (pointW 8aad5b3: EnvRobosuite
# stores the sim in self.env, no base_env, no experiment.logging) which CANNOT run NVlabs/
# mimicgen's datagen (it needs ARISE robomimic d0b37cf with base_env + experiment.logging).
# The two robomimic forks are mutually incompatible -> two envs.
#
# WHY THE HANDOFF IS SAFE: robosuite (b9d8d3de == v1.4.1) and mujoco (2.3.2) are IDENTICAL in
# both envs, so the datagen HDF5 (sim states + model xml) re-renders DETERMINISTICALLY in the
# equidiff env. Only robomimic differs.  (versions verified via workflow wlsj7ps2o, 2026-07-31)
#
#   cd <a workdir for the src checkouts> && bash /path/to/mea_diff/setup_datagen_env.sh
set -e
ENV=${1:-mimicgen_datagen}
ROBOSUITE=b9d8d3de5e3dfd1724f4a0e6555246c460407daa      # v1.4.1 — MUST match EquiDiff's robosuite
ROBOMIMIC=d0b37cf214bd24fb590d182edb6384333f67b661      # ARISE master — datagen-compatible API
TASKZOO=74eab7f88214c21ca1ae8617c2b2f8d19718a9ed

conda create -n "$ENV" python=3.8 -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$ENV"

pip install mujoco==2.3.2

git clone https://github.com/ARISE-Initiative/robosuite.git
( cd robosuite && git checkout "$ROBOSUITE" && pip install -e . )

git clone https://github.com/ARISE-Initiative/robomimic.git
( cd robomimic && git checkout "$ROBOMIMIC" && pip install -e . )

git clone https://github.com/ARISE-Initiative/robosuite-task-zoo.git
( cd robosuite-task-zoo && git checkout "$TASKZOO" && pip install -e . )

git clone https://github.com/NVlabs/mimicgen.git
( cd mimicgen && pip install -e . )

echo "== verify (mujoco 2.3.2, robosuite 1.4.x, robomimic 0.3.x, mimicgen) =="
pip list | grep -iE "^mujoco|^robosuite|^robomimic|^mimicgen"
echo "datagen env '$ENV' ready."
