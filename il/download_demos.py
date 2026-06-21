"""Fetch the WarehouseSort rgb demonstration datasets from the Kaggle competition into
il/demos/<level>/.

The demos (200 rgb episodes per level) are the competition data:
https://www.kaggle.com/competitions/marso-hack-berlin-2026-robot-parcel-sorting-challenge/data

  * On Kaggle: the competition data mounts under /kaggle/input/ — this finds it automatically.
  * Elsewhere: it downloads via `kagglehub` (you must have joined the competition and have a
    Kaggle API token — kaggle.com -> Settings -> Create New API Token, then put kaggle.json in
    ~/.kaggle/ or set KAGGLE_USERNAME / KAGGLE_KEY).

Either way the files are staged under il/demos/<level>/ so the trainer finds them.

  pixi run python il/download_demos.py
  pixi run python il/download_demos.py --competition <competition-slug>
  pixi run python il/download_demos.py --force   # re-download even if cached
"""

import argparse
import glob
import os
import shutil
import tarfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_COMPETITION = "marso-hack-berlin-2026-robot-parcel-sorting-challenge"
LEVELS = ("easy", "medium", "hard")
STATE_H5 = "trajectory.state.pd_ee_delta_pos.physx_cuda.h5"


def _levels_with_state_h5(root):
    """Return sorted difficulty folders that contain the state-track .h5."""
    return sorted(set(
        os.path.basename(os.path.dirname(p))
        for p in glob.glob(os.path.join(root, "**", STATE_H5), recursive=True)
    ))


def _find_mounted():
    """Return a Kaggle-mounted competition input dir, if any."""
    for p in glob.glob("/kaggle/input/*"):
        if (_levels_with_state_h5(p)
                or glob.glob(os.path.join(p, "**/*.tar.gz"), recursive=True)):
            print("found attached Kaggle competition data:", p)
            return p
    return None


def _download(competition, force=False):
    import kagglehub
    if force:
        print("re-downloading competition data (force_download=True)...")
    return kagglehub.competition_download(competition, force_download=force)


def _stage_from_src(src, dest):
    """Copy or extract demos from ``src`` into ``dest/<level>/``."""
    os.makedirs(dest, exist_ok=True)
    tars = glob.glob(os.path.join(src, "**/*.tar.gz"), recursive=True)
    if tars:
        for t in tars:
            with tarfile.open(t) as tf:
                tf.extractall(dest)
        return

    for h5 in glob.glob(os.path.join(src, "**/trajectory.*.pd_ee_delta_pos.physx_cuda.h5"), recursive=True):
        lvl = os.path.basename(os.path.dirname(h5))
        lvl_dir = os.path.join(dest, lvl)
        os.makedirs(lvl_dir, exist_ok=True)
        for f in glob.glob(h5[:-2] + "*"):   # the .h5 and its .json
            shutil.copy2(f, os.path.join(lvl_dir, os.path.basename(f)))


def fetch_and_stage(competition=DEFAULT_COMPETITION, dest=None, force=False):
    """Download (if needed) and stage easy/medium/hard demos. Returns staged level names."""
    dest = dest or os.path.join(REPO, "il", "demos")

    src = _find_mounted()
    if src is None:
        src = _download(competition, force=force)
        src_levels = _levels_with_state_h5(src)
        missing = set(LEVELS) - set(src_levels)
        if missing and not force:
            print(f"cached download incomplete (missing {sorted(missing)}); forcing fresh download...")
            src = _download(competition, force=True)

    print("data at:", src)
    _stage_from_src(src, dest)

    staged = _levels_with_state_h5(dest)
    print("staged levels:", staged)
    missing = set(LEVELS) - set(staged)
    if missing:
        raise RuntimeError(
            f"missing demo levels under {dest}: {sorted(missing)}. "
            "Join the competition on Kaggle and re-run with --force."
        )
    return staged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", default=DEFAULT_COMPETITION,
                    help="Kaggle competition slug")
    ap.add_argument("--dest", default=os.path.join(REPO, "il", "demos"))
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if kagglehub has a cached copy")
    args = ap.parse_args()

    fetch_and_stage(args.competition, dest=args.dest, force=args.force)
    got = sorted(glob.glob(os.path.join(args.dest, "*", "trajectory.*.pd_ee_delta_pos.physx_cuda.h5")))
    print(f"staged {len(got)} dataset file(s) under {args.dest}:")
    for f in got:
        print(" ", f)


if __name__ == "__main__":
    main()
