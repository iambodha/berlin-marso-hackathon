#!/usr/bin/env bash
# ManiSkill / Sapien environment for headless Thunder Compute GPUs.
# Must be sourced BEFORE launching Python (LD_LIBRARY_PATH is read at process start).
#
#   source scripts/thunder_env.sh

export DISPLAY="${DISPLAY:-}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

# Extra NVIDIA GL lib dirs saved by thunder_fix_vulkan.sh
if [[ -f "${HOME}/.vulkan/thunder_ld_path" ]]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$(head -1 "${HOME}/.vulkan/thunder_ld_path"):"*) ;;
    *)
      export LD_LIBRARY_PATH="$(head -1 "${HOME}/.vulkan/thunder_ld_path")${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      ;;
  esac
fi

_thunder_add_ld() {
  local dir="$1"
  [[ -d "$dir" ]] || return 0
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${dir}:"*) ;;
    *) export LD_LIBRARY_PATH="${dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
}

# libcuda.so — Sapien calls ctypes.CDLL("libcuda.so") for GPU PhysX.
if command -v ldconfig >/dev/null 2>&1; then
  while IFS= read -r lib; do
    _thunder_add_ld "$(dirname "$lib")"
  done < <(ldconfig -p 2>/dev/null | awk '/libcuda\.so/{print $NF}')
fi

for dir in \
  /usr/lib/x86_64-linux-gnu \
  /usr/lib64/nvidia \
  /usr/lib64 \
  /usr/local/nvidia/lib64 \
  /usr/local/cuda/lib64 \
  /usr/local/cuda/lib64/stubs \
  /usr/local/cuda/compat/lib; do
  if [[ -f "${dir}/libcuda.so" || -f "${dir}/libcuda.so.1" ]]; then
    _thunder_add_ld "$dir"
  fi
done

# Vulkan ICD (headless rendering)
if [[ -f "${HOME}/.vulkan/icd.d/nvidia_icd.json" ]]; then
  export VK_ICD_FILENAMES="${HOME}/.vulkan/icd.d/nvidia_icd.json"
else
  for icd in \
    /usr/share/vulkan/icd.d/nvidia_icd.json \
    /etc/vulkan/icd.d/nvidia_icd.json; do
    if [[ -f "$icd" ]]; then
      export VK_ICD_FILENAMES="$icd"
      break
    fi
  done
fi

if [[ -z "${VK_ICD_FILENAMES:-}" ]]; then
  echo "WARNING: No Vulkan ICD found. Eval/video needs Vulkan — run:  bash scripts/thunder_fix_vulkan.sh" >&2
fi

if ! (ldconfig -p 2>/dev/null | grep -q 'libcuda\.so' \
      || [[ -n "${LD_LIBRARY_PATH:-}" ]]); then
  echo "WARNING: libcuda.so not found on LD_LIBRARY_PATH." >&2
  echo "  Try:  find /usr -name 'libcuda.so*' 2>/dev/null" >&2
  echo "  Then:  export LD_LIBRARY_PATH=/path/to/dir:\$LD_LIBRARY_PATH" >&2
fi
