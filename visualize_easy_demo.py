#!/usr/bin/env python3
"""Render an easy-level demo episode as a video.

Shows the scene-camera and arm-camera RGB feeds alongside robot TCP pose, actions,
and privileged actor positions (parcels, bins) from the trajectory file.

If the trajectory was recorded before the arm camera existed, arm frames are
re-rendered from stored env states (requires ManiSkill + a GPU or CPU sim backend).

Example:
    python visualize_easy_demo.py
    python visualize_easy_demo.py --episode 3 --output media/easy_ep3.mp4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Headless rendering defaults (safe on macOS / Colab / servers).
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import h5py
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_H5 = REPO_ROOT / "il" / "demos" / "easy" / "trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5"


def _actor_xyz(actor_states: dict[str, np.ndarray], step: int) -> dict[str, np.ndarray]:
    return {name: actor_states[name][step, :3] for name in actor_states}


def _upsample_rgb(rgb: np.ndarray, scale: int = 3) -> np.ndarray:
    if scale == 1:
        return rgb
    return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)


def load_env_kwargs(h5_path: Path) -> dict:
    json_path = h5_path.with_suffix(".json")
    if not json_path.exists():
        return {}
    meta = json.loads(json_path.read_text())
    kwargs = dict(meta.get("env_info", {}).get("env_kwargs", {}))
    for key in ("render_mode", "obs_mode", "sim_backend", "num_envs", "reward_mode"):
        kwargs.pop(key, None)
    return kwargs


def load_episode(h5_path: Path, episode: int) -> dict:
    traj_key = f"traj_{episode}"
    with h5py.File(h5_path, "r") as f:
        if traj_key not in f:
            available = sorted(k for k in f.keys() if k.startswith("traj_"))
            raise SystemExit(
                f"{traj_key} not found in {h5_path}. "
                f"Available: {available[:5]}{'...' if len(available) > 5 else ''}"
            )
        traj = f[traj_key]
        actors = {
            name: traj["env_states"]["actors"][name][:]
            for name in traj["env_states"]["actors"].keys()
        }
        sensor_data = traj["obs"]["sensor_data"]
        arm_rgb = None
        if "arm_camera" in sensor_data:
            arm_rgb = sensor_data["arm_camera"]["rgb"][:]
        return {
            "scene_rgb": sensor_data["scene_camera"]["rgb"][:],
            "arm_rgb": arm_rgb,
            "tcp_pose": traj["obs"]["extra"]["tcp_pose"][:],
            "is_grasped": traj["obs"]["extra"]["is_grasped"][:],
            "qpos": traj["obs"]["agent"]["qpos"][:],
            "actions": traj["actions"][:],
            "actors": actors,
        }


def load_episode_meta(h5_path: Path, episode: int) -> dict | None:
    json_path = h5_path.with_suffix(".json")
    if not json_path.exists():
        return None
    meta = json.loads(json_path.read_text())
    for ep in meta.get("episodes", []):
        if ep.get("episode_id") == episode:
            return ep
    return None


def render_arm_camera_from_states(
    h5_path: Path,
    episode: int,
    n_steps: int,
    env_kwargs: dict,
    episode_meta: dict | None,
) -> np.ndarray | None:
    """Re-render wrist-camera frames by restoring recorded env states."""
    try:
        import gymnasium as gym
        import torch

        import warehouse_sort  # noqa: F401 — registers WarehouseSort-v1
        from mani_skill.trajectory import utils as trajectory_utils
    except ImportError:
        print("ManiSkill not installed — skipping arm camera re-render.")
        return None

    backend = "physx_cuda" if torch.cuda.is_available() else "physx_cpu"
    print(f"re-rendering arm camera via {backend} sim ...")

    with h5py.File(h5_path, "r") as f:
        traj = f[f"traj_{episode}"]
        env_states = trajectory_utils.dict_to_list_of_dicts(traj["env_states"])

    env = gym.make(
        "WarehouseSort-v1",
        num_envs=1,
        obs_mode="rgb",
        render_mode="rgb_array",
        sim_backend=backend,
        **env_kwargs,
    )
    unwrapped = env.unwrapped
    seed = episode_meta.get("episode_seed") if episode_meta else None
    env.reset(seed=seed)

    frames: list[np.ndarray] = []
    try:
        for step in range(n_steps):
            state = env_states[min(step, len(env_states) - 1)]
            unwrapped.set_state_dict(state)
            sensors = unwrapped.get_sensor_images()
            if "arm_camera" not in sensors:
                print("arm_camera sensor missing from env — update warehouse_sort/env.py first.")
                return None
            rgb = sensors["arm_camera"]["rgb"]
            frame = rgb[0].cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb[0])
            frames.append(frame)
            if (step + 1) % 20 == 0 or step + 1 == n_steps:
                print(f"arm camera re-rendered {step + 1}/{n_steps} frames")
    finally:
        env.close()
    return np.stack(frames, axis=0)


def _format_vec(values: np.ndarray, precision: int = 3) -> str:
    return "(" + ", ".join(f"{v:.{precision}f}" for v in values) + ")"


def render_frame(
    step: int,
    n_steps: int,
    scene_rgb: np.ndarray,
    arm_rgb: np.ndarray | None,
    tcp_pose: np.ndarray,
    is_grasped: bool,
    action: np.ndarray | None,
    actor_xyz: dict[str, np.ndarray],
    episode_meta: dict | None,
    rgb_scale: int,
) -> np.ndarray:
    fig = plt.figure(figsize=(16.0, 6.4), dpi=100)
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.1], wspace=0.08)

    ax_scene = fig.add_subplot(gs[0, 0])
    ax_scene.imshow(_upsample_rgb(scene_rgb, rgb_scale))
    ax_scene.set_title("Scene camera (policy view)", fontsize=11)
    ax_scene.axis("off")

    ax_arm = fig.add_subplot(gs[0, 1])
    if arm_rgb is not None:
        ax_arm.imshow(_upsample_rgb(arm_rgb, rgb_scale))
        ax_arm.set_title("Arm camera (wrist view)", fontsize=11)
    else:
        ax_arm.text(
            0.5,
            0.5,
            "Arm camera\nunavailable",
            ha="center",
            va="center",
            fontsize=12,
            color="0.4",
        )
        ax_arm.set_title("Arm camera (wrist view)", fontsize=11)
    ax_arm.axis("off")

    ax_map = fig.add_subplot(gs[0, 2])
    ax_map.set_title("Top-down positions (privileged)", fontsize=11)
    ax_map.set_xlabel("x (m)")
    ax_map.set_ylabel("y (m)")
    ax_map.set_aspect("equal")
    ax_map.grid(True, alpha=0.3)

    tcp_xyz = tcp_pose[:3]
    points: list[tuple[float, float, str, str]] = [
        (tcp_xyz[0], tcp_xyz[1], "tab:red", "TCP"),
    ]
    for name, color, label in [
        ("bin_red", "tab:red", "bin red"),
        ("bin_blue", "tab:blue", "bin blue"),
        ("parcel_0_env0", "tab:orange", "parcel 0"),
        ("parcel_1_env0", "tab:purple", "parcel 1"),
    ]:
        if name in actor_xyz:
            xyz = actor_xyz[name]
            points.append((xyz[0], xyz[1], color, label))

    for x, y, color, label in points:
        ax_map.scatter([x], [y], s=90, c=color, edgecolors="black", linewidths=0.5, zorder=3)
        ax_map.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    pad = 0.08
    ax_map.set_xlim(min(xs) - pad, max(xs) + pad)
    ax_map.set_ylim(min(ys) - pad, max(ys) + pad)

    title = f"Step {step + 1}/{n_steps}"
    if episode_meta:
        title += (
            f"  |  episode {episode_meta.get('episode_id')}  "
            f"seed {episode_meta.get('episode_seed')}  "
            f"success={episode_meta.get('success')}"
        )
    fig.suptitle(title, fontsize=12, y=0.98)

    action_text = "—"
    if action is not None:
        action_text = (
            f"Δxyz={_format_vec(action[:3])}  gripper={action[3]:+.2f} "
            f"({'open' if action[3] > 0 else 'close'})"
        )

    info_lines = [
        f"TCP position xyz: {_format_vec(tcp_pose[:3])}",
        f"TCP orientation quat: {_format_vec(tcp_pose[3:], precision=3)}",
        f"Grasped: {bool(is_grasped)}",
        f"Action: {action_text}",
        "",
        "Actor positions xyz:",
    ]
    for name in sorted(actor_xyz):
        info_lines.append(f"  {name}: {_format_vec(actor_xyz[name])}")

    fig.text(
        0.02,
        0.02,
        "\n".join(info_lines),
        fontsize=9,
        family="monospace",
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )

    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    image = buf[:, :, :3].copy()
    plt.close(fig)
    return image


def make_video(
    h5_path: Path,
    output: Path,
    episode: int,
    fps: int,
    max_steps: int | None,
    skip_arm_rerender: bool,
) -> Path:
    data = load_episode(h5_path, episode)
    meta = load_episode_meta(h5_path, episode)

    scene_rgb = data["scene_rgb"]
    arm_rgb = data["arm_rgb"]
    n_steps = len(scene_rgb)
    if max_steps is not None:
        n_steps = min(n_steps, max_steps)

    if arm_rgb is None and not skip_arm_rerender:
        env_kwargs = load_env_kwargs(h5_path)
        if env_kwargs:
            arm_rgb = render_arm_camera_from_states(
                h5_path, episode, n_steps, env_kwargs, meta
            )
        else:
            print("No trajectory .json found — cannot re-render arm camera.")

    if arm_rgb is not None:
        arm_rgb = arm_rgb[:n_steps]

    output.parent.mkdir(parents=True, exist_ok=True)
    frames: list[np.ndarray] = []
    for step in range(n_steps):
        action = data["actions"][step] if step < len(data["actions"]) else None
        actor_xyz = _actor_xyz(data["actors"], step)
        arm_frame = arm_rgb[step] if arm_rgb is not None else None
        frame = render_frame(
            step=step,
            n_steps=n_steps,
            scene_rgb=scene_rgb[step],
            arm_rgb=arm_frame,
            tcp_pose=data["tcp_pose"][step],
            is_grasped=bool(data["is_grasped"][step]),
            action=action,
            actor_xyz=actor_xyz,
            episode_meta=meta,
            rgb_scale=3,
        )
        frames.append(frame)
        if (step + 1) % 20 == 0 or step + 1 == n_steps:
            print(f"rendered {step + 1}/{n_steps} frames")

    iio.imwrite(output, frames, fps=fps, codec="libx264", quality=8)
    print(f"saved {output} ({len(frames)} frames @ {fps} fps)")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5, help="Path to easy demo .h5 file")
    parser.add_argument("--episode", type=int, default=0, help="Episode index (traj_<episode>)")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "media" / "easy_demo_ep0.mp4",
        help="Output video path",
    )
    parser.add_argument("--fps", type=int, default=10, help="Output video frame rate")
    parser.add_argument("--max-steps", type=int, default=None, help="Limit frames (for quick previews)")
    parser.add_argument(
        "--skip-arm-rerender",
        action="store_true",
        help="Do not re-render arm camera from env states when missing from the .h5",
    )
    args = parser.parse_args()

    if not args.h5.exists():
        raise SystemExit(f"Demo file not found: {args.h5}\nRun download_kaggle_data.py first.")

    make_video(
        args.h5,
        args.output,
        args.episode,
        args.fps,
        args.max_steps,
        args.skip_arm_rerender,
    )


if __name__ == "__main__":
    main()
