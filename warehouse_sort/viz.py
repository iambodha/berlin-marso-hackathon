"""Lightweight state observation plots (works on Windows without Vulkan offscreen)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

# Match WarehouseSortEnv geometry (meters).
PARCEL_HALF = (0.026, 0.026, 0.03)
BIN_HALF = (0.11, 0.13)
BIN_WALL_H = 0.025
BIN_FLOOR_T = 0.005
INBOUND_HALF = (0.10, 0.12)
TAG_COLORS = ("tab:red", "tab:blue")


def _as_numpy(obs):
    if hasattr(obs, "detach"):
        obs = obs[0].detach().cpu().numpy()
    else:
        obs = np.asarray(obs)[0]
    return obs


def _current_state_vector(obs) -> np.ndarray:
    """Return the latest (obs_dim,) vector from a step or FrameStacked observation."""
    s = _as_numpy(obs)
    if s.ndim == 2:
        return s[-1]
    return s


@dataclass
class ParsedState:
    tcp_xyz: np.ndarray
    tcp_quat: np.ndarray
    is_grasped: float
    parcel_xyz: np.ndarray
    parcel_quat: np.ndarray
    parcel_tag: np.ndarray
    bin_xyz: np.ndarray


def parse_state_obs(obs, num_parcels: int) -> ParsedState:
    """Decode the privileged state vector into 3D scene elements."""
    s = _current_state_vector(obs)
    p = int(num_parcels)
    base = 26
    parcel_block = s[base : base + p * 7].reshape(p, 7)
    tag = s[base + p * 7 : base + p * 7 + p * 2].reshape(p, 2)
    bins = s[base + p * 7 + p * 2 : base + p * 7 + p * 2 + 6].reshape(2, 3)
    return ParsedState(
        tcp_xyz=s[18:21],
        tcp_quat=s[21:25],
        is_grasped=float(s[25]),
        parcel_xyz=parcel_block[:, :3],
        parcel_quat=parcel_block[:, 3:7],
        parcel_tag=tag,
        bin_xyz=bins,
    )


def _quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def _wireframe_box(ax, center, half_sizes, color, quat=None, linewidth=1.0, alpha=0.95):
    hx, hy, hz = half_sizes
    corners = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ], dtype=float)
    if quat is not None:
        corners = (corners @ _quat_wxyz_to_rot(quat).T) + center
    else:
        corners = corners + center
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    for i, j in edges:
        xs, ys, zs = corners[[i, j], 0], corners[[i, j], 1], corners[[i, j], 2]
        ax.plot(xs, ys, zs, color=color, linewidth=linewidth, alpha=alpha)


def _filled_rect_3d(ax, center_xy, half_xy, z, color, alpha=0.25):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    cx, cy = center_xy
    hx, hy = half_xy
    verts = [[
        [cx - hx, cy - hy, z],
        [cx + hx, cy - hy, z],
        [cx + hx, cy + hy, z],
        [cx - hx, cy + hy, z],
    ]]
    ax.add_collection3d(
        Poly3DCollection(verts, facecolors=color, alpha=alpha, edgecolors=color, linewidths=0.8)
    )


def plot_state_3d(
    parsed: ParsedState,
    ax=None,
    title: str = "Rollout (state obs, schematic 3D)",
    step: Optional[int] = None,
    limits=((-0.35, 0.35), (-0.2, 0.5), (0.0, 0.35)),
    elev: float = 22.0,
    azim: float = -58.0,
):
    """Schematic 3D scene from parsed state (no Vulkan / SAPIEN renderer)."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")

    (xlim, ylim, zlim) = limits
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=elev, azim=azim)

    head = title if step is None else f"{title}  step {step}"
    ax.set_title(head, fontsize=10)

    _filled_rect_3d(ax, (0.0, 0.0), INBOUND_HALF, z=0.002, color="0.75", alpha=0.35)
    for i, pos in enumerate(parsed.bin_xyz):
        c = TAG_COLORS[i]
        floor_z = pos[2] + BIN_FLOOR_T
        _wireframe_box(
            ax, pos + np.array([0.0, 0.0, BIN_FLOOR_T + BIN_WALL_H / 2]),
            (BIN_HALF[0], BIN_HALF[1], BIN_WALL_H / 2), color=c, linewidth=1.4,
        )
        _filled_rect_3d(ax, pos[:2], BIN_HALF, z=floor_z, color=c, alpha=0.12)

    for i, (pos, quat, tag) in enumerate(
        zip(parsed.parcel_xyz, parsed.parcel_quat, parsed.parcel_tag)
    ):
        c = TAG_COLORS[int(np.argmax(tag))]
        _wireframe_box(ax, pos, PARCEL_HALF, color=c, quat=quat, linewidth=1.6)

    tcp = parsed.tcp_xyz
    ax.scatter([tcp[0]], [tcp[1]], [tcp[2]], s=80, c="gold", edgecolors="k", depthshade=True)
    if parsed.is_grasped > 0.5:
        ax.plot([tcp[0]], [tcp[1]], [tcp[2]], marker="x", color="k", markersize=8)

    return ax


def save_state_rollout_video(
    parsed_frames: Sequence[ParsedState],
    out_path: str,
    fps: int = 20,
    title: str = "Policy rollout (schematic 3D)",
    dpi: int = 100,
) -> str:
    """Encode a list of parsed states to gif/mp4 using matplotlib (Windows-safe)."""
    import matplotlib.pyplot as plt
    from matplotlib import animation

    if not parsed_frames:
        raise ValueError("No frames to render")

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    def _draw(i):
        ax.clear()
        plot_state_3d(parsed_frames[i], ax=ax, title=title, step=i)

    ani = animation.FuncAnimation(
        fig, _draw, frames=len(parsed_frames), interval=max(1, int(1000 / fps))
    )

    out_path = str(out_path)
    if out_path.lower().endswith(".mp4"):
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            ani.save(out_path, writer=writer, dpi=dpi)
            plt.close(fig)
            return out_path
        except Exception:
            out_path = out_path[:-4] + ".gif"

    writer = animation.PillowWriter(fps=fps)
    ani.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_state_topdown(obs, num_parcels: int = 2, ax=None, title: str = "Top-down (from state obs)"):
    """Draw parcel / bin / TCP positions from the privileged state vector."""
    import matplotlib.pyplot as plt

    s = _as_numpy(obs)
    p = int(num_parcels)
    tcp = s[18:21]
    parcel_block = s[26 : 26 + p * 7].reshape(p, 7)[:, :3]
    tag = s[26 + p * 7 : 26 + p * 7 + p * 2].reshape(p, 2)
    bins = s[26 + p * 7 + p * 2 : 26 + p * 7 + p * 2 + 6].reshape(2, 3)
    tag_colors = ["tab:red", "tab:blue"]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    bx, by = 0.08, 0.08
    for i, (pos, color_idx) in enumerate(zip(bins, (0, 1))):
        c = tag_colors[color_idx]
        ax.add_patch(plt.Rectangle((pos[0] - bx, pos[1] - by), 2 * bx, 2 * by,
                                   fill=False, ec=c, lw=2, label=f"bin {i}" if i == 0 else None))
    for i, pos in enumerate(parcel_block):
        c = tag_colors[int(np.argmax(tag[i]))]
        ax.scatter(pos[0], pos[1], s=120, c=c, edgecolors="k", zorder=3, label=f"parcel {i}")
    ax.scatter(tcp[0], tcp[1], s=160, c="gold", marker="*", edgecolors="k", zorder=4, label="TCP")
    ax.legend(loc="upper right", fontsize=8)
    return ax
