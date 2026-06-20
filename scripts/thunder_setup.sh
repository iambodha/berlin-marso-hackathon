#!/usr/bin/env bash
# One-time setup on a fresh Thunder Compute GPU instance (Ubuntu + CUDA).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> WarehouseSort setup in $ROOT"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Create a GPU instance first (tnr create)." >&2
  exit 1
fi
nvidia-smi || true

# ManiSkill / Sapien headless rendering deps (libX11, Vulkan, EGL).
if command -v apt-get >/dev/null 2>&1; then
  echo "==> Installing system libraries for Sapien/ManiSkill..."
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    libx11-6 libxext6 libxrender1 libxi6 libxrandr2 libxinerama1 libxcursor1 \
    libgl1 libegl1 libvulkan1 vulkan-tools mesa-vulkan-drivers \
    libglib2.0-0 libgomp1
  sudo ldconfig
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -U pip wheel

# PyTorch with CUDA — use versions available on download.pytorch.org (2.12.x is not published).
install_torch() {
  local index="$1"
  python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url "$index"
}

if ! install_torch "https://download.pytorch.org/whl/cu124"; then
  echo "cu124 wheels failed — trying cu121..."
  install_torch "https://download.pytorch.org/whl/cu121"
fi

python -m pip install -r requirements-thunder.txt
python -m pip install -e .

export DISPLAY=""
export PYOPENGL_PLATFORM=egl
export HDF5_USE_FILE_LOCKING=FALSE
# shellcheck disable=SC1091
source "$(dirname "$0")/thunder_env.sh"

python - <<'PY'
import warehouse_sort  # registers env
import gymnasium as gym
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

env = gym.make(
    "WarehouseSort-v1", num_envs=1, obs_mode="rgb",
    control_mode="pd_ee_delta_pos", sim_backend="gpu", render_mode="rgb_array",
    difficulty="easy", num_parcels=2, fixed_poses=True,
)
env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
env.reset(seed=0)
env.close()
print("ManiSkill / Sapien import OK")
PY

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo
echo "Setup complete. Next:"
echo "  source .venv/bin/activate"
echo "  python run_pipeline.py"
echo
if ! ldconfig -p 2>/dev/null | grep -q 'libEGL_nvidia.so'; then
  echo "NOTE: libEGL_nvidia not found — eval/video needs Vulkan."
  echo "  Thunder Development mode is CUDA-only. For eval on Thunder:"
  echo "    tnr modify 0 --mode production --num-gpus 1 -y   # on your Mac"
  echo "  Then: bash scripts/thunder_fix_vulkan.sh"
  echo "  Or pull checkpoints and eval on Colab — see THUNDER_COMPUTE.md"
fi
