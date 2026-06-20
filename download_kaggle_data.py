#!/usr/bin/env python3
"""Download the WarehouseSort rgb demonstration datasets from Kaggle.

Competition:
  https://www.kaggle.com/competitions/marso-hack-berlin-2026-robot-parcel-sorting-challenge/data

Before running:
  1. Join the competition (Rules -> "I Understand and Accept").
  2. Create an API token: kaggle.com -> avatar -> Settings -> API -> Create New API Token.
     This downloads kaggle.json with your username and key.
  3. Paste those values below, then run:

       pip install kaggle
       python download_kaggle_data.py

Files are staged under il/demos/<level>/ (easy, medium, hard).
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

# ============ PASTE YOUR KAGGLE CREDENTIALS HERE ============
KAGGLE_USERNAME = "unrealdrip"
KAGGLE_KEY = "412081ee9034f8cc6665fe2560ecb92b"
# ==============================================================

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DEST = REPO_ROOT / "il" / "demos"
DEFAULT_COMPETITION = "marso-hack-berlin-2026-robot-parcel-sorting-challenge"


def _apply_credentials(username: str | None, key: str | None) -> None:
    resolved_username = username or os.environ.get("KAGGLE_USERNAME") or KAGGLE_USERNAME
    resolved_key = key or os.environ.get("KAGGLE_KEY") or KAGGLE_KEY
    if not resolved_username or not resolved_key:
        raise SystemExit(
            "Missing Kaggle credentials.\n"
            "Paste KAGGLE_USERNAME and KAGGLE_KEY at the top of download_kaggle_data.py,\n"
            "or pass --username / --key, or set the KAGGLE_USERNAME / KAGGLE_KEY env vars."
        )
    os.environ["KAGGLE_USERNAME"] = resolved_username
    os.environ["KAGGLE_KEY"] = resolved_key


def _extract_archives(root: Path) -> None:
    for archive in sorted(root.glob("*.zip")):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(root)
        archive.unlink()

    for archive in list(root.rglob("*.tar.gz")):
        with tarfile.open(archive) as tf:
            tf.extractall(root)
        archive.unlink()


def download_competition(competition: str, download_dir: Path) -> Path:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit(
            "The 'kaggle' package is required. Install it with:\n\n  pip install kaggle\n"
        ) from exc

    download_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"Downloading competition data: {competition}")
    api.competition_download_files(competition, path=str(download_dir), quiet=False)
    _extract_archives(download_dir)
    return download_dir


def stage_demos(src: Path, dest: Path) -> list[Path]:
    """Move demo files into il/demos/<level>/."""
    dest.mkdir(parents=True, exist_ok=True)

    tars = glob.glob(str(src / "**" / "*.tar.gz"), recursive=True)
    if tars:
        for tar_path in tars:
            with tarfile.open(tar_path) as tf:
                tf.extractall(dest)
            os.remove(tar_path)
    else:
        for h5 in glob.glob(str(src / "**" / "trajectory.rgb.*.h5"), recursive=True):
            level = os.path.basename(os.path.dirname(h5))
            level_dir = dest / level
            level_dir.mkdir(parents=True, exist_ok=True)
            for companion in glob.glob(h5[:-2] + "*"):
                target = level_dir / os.path.basename(companion)
                if target.exists():
                    target.unlink()
                shutil.move(companion, target)

    datasets = sorted(dest.glob("*/trajectory.rgb.*.h5"))
    if not datasets:
        raise SystemExit(
            f"No trajectory.rgb.*.h5 files found under {src}.\n"
            "Make sure you joined the competition and your credentials are valid."
        )
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--competition",
        default=DEFAULT_COMPETITION,
        help="Kaggle competition slug",
    )
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_DEST),
        help="Destination directory for staged demos (default: il/demos/)",
    )
    parser.add_argument("--username", help="Kaggle username (overrides script/env)")
    parser.add_argument("--key", help="Kaggle API key (overrides script/env)")
    args = parser.parse_args()

    _apply_credentials(args.username, args.key)
    dest = Path(args.dest)
    cache = dest / ".kaggle_cache"
    cache.mkdir(parents=True, exist_ok=True)

    try:
        src = download_competition(args.competition, cache)
        print("Downloaded to:", src)
        datasets = stage_demos(src, dest)
    finally:
        shutil.rmtree(cache, ignore_errors=True)

    print(f"Staged {len(datasets)} level dataset(s) under {dest}:")
    for dataset in datasets:
        print(f"  {dataset}")


if __name__ == "__main__":
    main()
