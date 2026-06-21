"""Inject the privileged scene state into the RGB demo datasets (image + state track).

The organisers enabled using state (object poses etc.) ALONGSIDE the image. The shipped rgb
demos only stored 26-dim proprioception, but every trajectory already carries the full
``env_states`` (parcel + bin poses). This script reads those env states and writes the same
privileged fields the env now exposes in ``_get_obs_extra`` into ``obs/extra`` of a new
dataset, so training data matches the (patched) eval observation exactly -- no slow re-replay,
no re-rendering of images.

Parcel slots are zero-padded to ``MAX_PARCELS`` (matching ``WarehouseSortEnv.MAX_PARCELS``) so
the state vector is a FIXED size at every difficulty -> one general image+state policy.

Output (per chosen output dir):
    il/demos/<out>/trajectory.rgbstate.pd_ee_delta_pos.physx_cuda.h5   (+ .json)

Usage:
    # single level (honours "train on medium")
    pixi run python il/add_state_to_rgb_demos.py --levels medium

    # merged across all levels -> a single general dataset (recommended for the weighted score,
    # since the model then sees 2/4/6 active parcels, not just medium's 4)
    pixi run python il/add_state_to_rgb_demos.py --levels easy medium hard --out combined
"""

import argparse
import json
import os
import shutil

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_PARCELS = 6  # keep in sync with WarehouseSortEnv.MAX_PARCELS

RGB_STEM = "trajectory.rgb.pd_ee_delta_pos.physx_cuda"
OUT_STEM = "trajectory.rgbstate.pd_ee_delta_pos.physx_cuda"


def _parcel_tag_onehot(P):
    """tags = [i % 2] (red=0 -> [1,0], blue=1 -> [0,1]); matches env.parcel_tags."""
    oh = np.zeros((P, 2), np.float32)
    for i in range(P):
        oh[i, i % 2] = 1.0
    return oh.reshape(-1)  # (P*2,)


def _build_extra(traj, P):
    """Reconstruct the padded privileged extra fields from a trajectory's env_states."""
    T = traj["obs/agent/qpos"].shape[0]
    actors = traj["env_states/actors"]

    poses = [actors[f"parcel_{j}_env0"][:, :7] for j in range(P)]   # each (T, 7)
    parcel_pose = np.concatenate(poses, axis=1).astype(np.float32)  # (T, P*7)
    parcel_tag = np.tile(_parcel_tag_onehot(P), (T, 1)).astype(np.float32)  # (T, P*2)
    if P < MAX_PARCELS:
        pad = MAX_PARCELS - P
        parcel_pose = np.concatenate([parcel_pose, np.zeros((T, pad * 7), np.float32)], axis=1)
        parcel_tag = np.concatenate([parcel_tag, np.zeros((T, pad * 2), np.float32)], axis=1)

    bin_position = np.concatenate(
        [actors["bin_red"][:, :3], actors["bin_blue"][:, :3]], axis=1
    ).astype(np.float32)  # (T, 6)
    bin_color = np.tile(np.array([1, 0, 0, 1], np.float32), (T, 1))  # (T, 4)
    return dict(
        parcel_pose=parcel_pose,
        parcel_tag=parcel_tag,
        bin_position=bin_position,
        bin_color=bin_color,
    )


def _level_paths(level):
    d = os.path.join(REPO, "il", "demos", level)
    return os.path.join(d, RGB_STEM + ".h5"), os.path.join(d, RGB_STEM + ".json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", default=["medium"],
                    choices=["easy", "medium", "hard"],
                    help="which level(s) to convert; multiple -> merged into one dataset")
    ap.add_argument("--out", default=None,
                    help="output dir name under il/demos/ (default: the level name, or "
                         "'combined' when merging several levels)")
    args = ap.parse_args()

    out_name = args.out or (args.levels[0] if len(args.levels) == 1 else "combined")
    out_dir = os.path.join(REPO, "il", "demos", out_name)
    os.makedirs(out_dir, exist_ok=True)
    out_h5 = os.path.join(out_dir, OUT_STEM + ".h5")
    out_json = os.path.join(out_dir, OUT_STEM + ".json")

    merged_episodes = []
    env_info_ref = None
    max_P = 0
    n_written = 0

    with h5py.File(out_h5, "w") as dst:
        for level in args.levels:
            src_h5, src_json = _level_paths(level)
            if not os.path.exists(src_h5):
                raise FileNotFoundError(f"missing rgb demos for '{level}': {src_h5}\n"
                                        "  run il/download_demos.py first")
            info = json.load(open(src_json))
            P = int(info["env_info"]["env_kwargs"]["num_parcels"])
            max_P = max(max_P, P)
            if P > MAX_PARCELS:
                raise ValueError(f"level {level} has {P} parcels > MAX_PARCELS={MAX_PARCELS}")
            if env_info_ref is None or P == max_P:
                env_info_ref = info["env_info"]  # keep the largest-parcel level's env kwargs

            print(f"[{level}] P={P}  reading {src_h5}", flush=True)
            with h5py.File(src_h5, "r") as src:
                traj_keys = sorted(src.keys(), key=lambda x: int(x.split("_")[-1]))
                for tk in traj_keys:
                    new_name = f"traj_{n_written}"
                    src.copy(tk, dst, name=new_name)             # deep-copy the trajectory
                    extra = _build_extra(src[tk], P)
                    grp = dst[new_name]["obs/extra"]
                    for k, v in extra.items():
                        if k in grp:
                            del grp[k]
                        grp.create_dataset(k, data=v)
                    n_written += 1

            for ep in info["episodes"]:
                ep = dict(ep)
                ep["episode_id"] = len(merged_episodes)
                merged_episodes.append(ep)
            print(f"[{level}] copied {len(traj_keys)} trajectories", flush=True)

    # Companion json: env kwargs drive the periodic-eval env during training. Use the
    # largest-parcel level so that eval env emits the same fixed-size (padded) state.
    out_info = {
        "env_info": env_info_ref,
        "episodes": merged_episodes,
    }
    out_info["env_info"]["env_kwargs"]["num_parcels"] = max_P
    json.dump(out_info, open(out_json, "w"))

    state_dim = 18 + 8 + MAX_PARCELS * 7 + MAX_PARCELS * 2 + 6 + 4
    print(f"\n[done] wrote {n_written} trajectories -> {out_h5}")
    print(f"[done] companion json -> {out_json}")
    print(f"[done] fixed state dim per step = {state_dim}  (proprio 26 + padded privileged)")
    print(f"\nTrain on it with:\n"
          f"  pixi run python il/train.py method=dp_rgb demo_dir={out_name}")


if __name__ == "__main__":
    main()
