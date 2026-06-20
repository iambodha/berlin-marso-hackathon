#!/usr/bin/env bash
# Fix headless Vulkan on cloud GPU instances (Thunder, etc.) for ManiSkill eval.
#
# Run once on the instance:
#   bash scripts/thunder_fix_vulkan.sh
#
# Then:
#   source scripts/thunder_env.sh
#   python eval.py ...
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

_thunder_add_ld() {
  local dir="$1"
  [[ -d "$dir" ]] || return 0
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${dir}:"*) ;;
    *) export LD_LIBRARY_PATH="${dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
}

search_lib() {
  local name="$1"
  local hit
  hit="$(ldconfig -p 2>/dev/null | awk -v n="$name" '$1 ~ n {print $NF; exit}')"
  if [[ -n "$hit" && -f "$hit" ]]; then
    echo "$hit"
    return 0
  fi
  hit="$(find /usr /lib /opt /usr/local 2>/dev/null \
    \( -path '/usr/local/cuda*' -o -path '/proc/*' \) -prune -o \
    -name "${name}" -type f -print 2>/dev/null | head -1)"
  if [[ -n "$hit" && -f "$hit" ]]; then
    echo "$hit"
    return 0
  fi
  return 1
}

echo "==> Installing Vulkan / GLVND packages..."
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    libvulkan1 vulkan-tools libegl1 libglvnd0 libgl1 libsm6 libxext6 \
    mesa-vulkan-drivers 2>/dev/null || true

  if command -v nvidia-smi >/dev/null 2>&1; then
    drv_major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
    echo "==> NVIDIA driver major version: ${drv_major}"
    for pkg in \
      "libnvidia-gl-${drv_major}" \
      "libnvidia-egl-${drv_major}" \
      nvidia-vulkan-icd \
      libnvidia-gl-550 libnvidia-gl-535 libnvidia-gl-525; do
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$pkg" 2>/dev/null || true
    done
  fi
  sudo ldconfig
fi

# Cloud images often put user-space GL libs outside the default loader path.
for dir in \
  /usr/lib/x86_64-linux-gnu \
  /usr/lib64/nvidia \
  /usr/lib64 \
  /usr/local/nvidia/lib64 \
  /usr/local/nvidia/lib \
  /lib/x86_64-linux-gnu; do
  if [[ -f "${dir}/libEGL_nvidia.so.0" || -f "${dir}/libGLX_nvidia.so.0" ]]; then
    _thunder_add_ld "$dir"
  fi
done

EGL_LIB="$(search_lib libEGL_nvidia.so.0 || true)"
GLX_LIB="$(search_lib libGLX_nvidia.so.0 || true)"

if [[ -z "$EGL_LIB" && -z "$GLX_LIB" ]]; then
  echo
  echo "ERROR: NVIDIA Vulkan/GL libraries not found on this instance." >&2
  echo "  Training works (CUDA only), but ManiSkill eval needs libEGL_nvidia / libGLX_nvidia." >&2
  echo >&2
  echo "Thunder Compute: Development (Prototyping) mode does NOT ship Vulkan/graphics libs." >&2
  echo "  Switch to Production on your Mac, then re-run this script:" >&2
  echo "    tnr modify 0 --mode production --num-gpus 1 -y" >&2
  echo >&2
  echo "Diagnostics:" >&2
  echo "  nvidia-smi" >&2
  echo "  find /usr /lib /opt -name 'libEGL_nvidia.so*' 2>/dev/null" >&2
  echo >&2
  echo "Alternatives:" >&2
  echo "  • After switching to production: re-run  bash scripts/thunder_fix_vulkan.sh" >&2
  echo "  • Stay in dev mode: train here, then on Mac run  bash scripts/thunder_sync.sh pull" >&2
  echo "    and eval on Google Colab (starter.ipynb)" >&2
  exit 1
fi

# GLVND EGL vendor file (required for headless EGL on many setups).
if [[ -n "$EGL_LIB" && ! -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]]; then
  echo "==> Installing GLVND EGL vendor config..."
  sudo mkdir -p /usr/share/glvnd/egl_vendor.d
  sudo tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json >/dev/null <<EOF
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "${EGL_LIB}"
    }
}
EOF
fi

ICD_DIR="${HOME}/.vulkan/icd.d"
mkdir -p "$ICD_DIR"
ICD_FILE="${ICD_DIR}/nvidia_icd.json"

# Headless: prefer EGL; fall back to GLX with absolute path.
if [[ -n "$EGL_LIB" ]]; then
  LIB_PATH="$EGL_LIB"
elif [[ -n "$GLX_LIB" ]]; then
  LIB_PATH="$GLX_LIB"
fi

cat > "$ICD_FILE" <<EOF
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "${LIB_PATH}",
        "api_version": "1.3.0"
    }
}
EOF
echo "==> Wrote ICD: $ICD_FILE"
echo "    library_path: $LIB_PATH"

# A100 / datacenter GPUs may need the Optimus layer.
if [[ ! -f /etc/vulkan/implicit_layer.d/nvidia_layers.json && -n "$GLX_LIB" ]]; then
  echo "==> Installing VK_LAYER_NV_optimus (some datacenter GPUs)..."
  sudo mkdir -p /etc/vulkan/implicit_layer.d
  sudo tee /etc/vulkan/implicit_layer.d/nvidia_layers.json >/dev/null <<EOF
{
    "file_format_version": "1.0.0",
    "layer": {
        "name": "VK_LAYER_NV_optimus",
        "type": "INSTANCE",
        "library_path": "${GLX_LIB}",
        "api_version": "1.3.0",
        "implementation_version": "1",
        "description": "NVIDIA Optimus layer",
        "functions": {
            "vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",
            "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"
        },
        "enable_environment": {
            "__NV_PRIME_RENDER_OFFLOAD": "1"
        },
        "disable_environment": {
            "DISABLE_LAYER_NV_OPTIMUS_1": ""
        }
    }
}
EOF
fi

export DISPLAY=""
export VK_ICD_FILENAMES="$ICD_FILE"

# Persist LD_LIBRARY_PATH hint for thunder_env.sh
ENV_HINT="${HOME}/.vulkan/thunder_ld_path"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  echo "$LD_LIBRARY_PATH" > "$ENV_HINT"
fi

echo
echo "==> Vulkan test (expect DISCRETE_GPU + your NVIDIA GPU name):"
if command -v vulkaninfo >/dev/null 2>&1; then
  if ! vulkaninfo --summary 2>&1 | tee /tmp/vulkan_test.log | grep -A8 "GPU0:"; then
    tail -25 /tmp/vulkan_test.log
    echo
    echo "Vulkan test failed — see errors above." >&2
    exit 1
  fi
else
  echo "vulkaninfo not installed"
fi

echo
echo "Done. Before eval, run:"
echo "  source scripts/thunder_env.sh"
echo "  python eval.py difficulty=easy policy=warehouse_sort.il_policy:load_dp_rgb \\"
echo "    checkpoint=il/baselines/diffusion_policy/runs/warehouse_rgb_dp/checkpoints/10000.pt \\"
echo "    eval_config=conf/eval/default.yaml"
