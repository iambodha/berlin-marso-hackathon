"""Windows-only: make SAPIEN GPU sim find cuda.dll before PhysX loads."""

from __future__ import annotations

import os
import platform
import shutil


def setup_windows_cuda() -> None:
    if platform.system() != "Windows":
        return

    cuda_root = os.environ.get(
        "CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
    )
    cuda_bin = os.path.join(cuda_root, "bin", "x64")

    # SAPIEN loads "cuda.dll"; on Windows the driver API is nvcuda.dll in System32.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    shim_dir = os.path.join(repo_root, ".cuda_shim")
    os.makedirs(shim_dir, exist_ok=True)
    shim_dll = os.path.join(shim_dir, "cuda.dll")
    if not os.path.exists(shim_dll):
        nvcuda = os.path.join(
            os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvcuda.dll"
        )
        shutil.copy2(nvcuda, shim_dll)

    os.add_dll_directory(shim_dir)
    if os.path.isdir(cuda_bin):
        os.add_dll_directory(cuda_bin)
    os.environ["PATH"] = (
        shim_dir + os.pathsep + cuda_bin + os.pathsep + os.environ.get("PATH", "")
    )


setup_windows_cuda()
