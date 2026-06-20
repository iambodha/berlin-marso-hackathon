"""Warehouse Colour-Sort Pick-and-Place environment (ManiSkill 3).

A Franka Panda picks brown cardboard parcels and places each into the bin whose colour
matches the parcel's top-face tag (red tag → red bin, blue tag → blue bin).
"""

from typing import Any, Optional

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat

TAG_BASE_COLORS = [
    [0.80, 0.10, 0.10],  # 0: red
    [0.10, 0.20, 0.80],  # 1: blue
]
BIN_BASE_COLORS = [
    [0.85, 0.15, 0.15],  # 0: red bin
    [0.15, 0.25, 0.85],  # 1: blue bin
]
CARDBOARD_BASE = [0.62, 0.46, 0.30]


def _to_list(x):
    if x is None:
        return None
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


@register_env("WarehouseSort-v1", max_episode_steps=100)
class WarehouseSortEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda"]
    SUPPORTED_REWARD_MODES = ["sparse", "none"]

    # 0.052 m box: largest that fits the gripper (~0.08 m) after max yaw rotation
    parcel_half = (0.026, 0.026, 0.03)
    tag_half    = (0.010, 0.010, 0.0015)
    inbound_half   = (0.10, 0.12)
    inbound_center = (0.0, 0.0)
    bin_half    = (0.11, 0.13)
    bin_wall_h  = 0.025
    bin_wall_t  = 0.008
    bin_floor_t = 0.005
    bin_base_x  = 0.0
    bin_base_y  = 0.36

    START_QPOS = [0.0, 0.3927, 0.0, -1.9635, 0.0, 2.356, 0.7854, 0.04, 0.04]

    def __init__(
        self,
        *args,
        difficulty: str = "easy",
        num_parcels: int = 2,
        fixed_poses: bool = True,
        randomization: Optional[dict] = None,
        camera_width: int = 128,
        camera_height: int = 128,
        robot_init_qpos_noise: float = 0.02,
        robot_uids: Optional[str] = None,
        obs_camera: str = "scene",  # kept for demo-replay compat; value is ignored
        max_episode_steps: int = 100,
        **kwargs,
    ):
        self.max_episode_steps = int(max_episode_steps)
        self.difficulty = difficulty
        self.num_parcels = int(num_parcels)
        self.fixed_poses = bool(fixed_poses)
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self._rand = self._normalize_rand(randomization)
        tags = [i % 2 for i in range(self.num_parcels)]
        self._parcel_tag_ids = tags
        if robot_uids is None:
            robot_uids = "panda"
        sensor_configs = kwargs.pop("sensor_configs", {}) or {}
        sensor_configs = {
            **{
                "scene_camera": dict(width=self.camera_width, height=self.camera_height),
                "arm_camera": dict(width=self.camera_width, height=self.camera_height),
            },
            **sensor_configs,
        }
        super().__init__(*args, robot_uids=robot_uids, sensor_configs=sensor_configs, **kwargs)

    @staticmethod
    def _normalize_rand(r):
        """Fill missing axes with zero/deterministic defaults."""
        r = dict(r) if r else {}

        def g(d, k, default):
            d = r.get(d, {}) or {}
            return d.get(k, default)

        return dict(
            parcel_xy_jitter=_to_list(g("parcel_pose", "xy_jitter", [0.0, 0.0])),
            parcel_yaw_jitter=_to_list(g("parcel_pose", "yaw_jitter", [0.0, 0.0])),
            bin_side_swap_prob=float(g("bin_position", "side_swap_prob", 0.0)),
            bin_xy_jitter=_to_list(g("bin_position", "xy_jitter", [0.0, 0.0])),
            light_intensity=_to_list(g("lighting", "intensity", [1.0, 1.0])),
            light_dir_jitter=_to_list(g("lighting", "direction_jitter", [0.0, 0.0])),
            table_colors=_to_list(g("background", "table_colors", [[0.3, 0.3, 0.3]])),
            floor_colors=_to_list(g("background", "floor_colors", [[0.2, 0.2, 0.2]])),
            cardboard_shade=_to_list(g("appearance", "cardboard_shade", [0.0, 0.0])),
            tag_shade=_to_list(g("appearance", "tag_shade", [0.0, 0.0])),
            clutter_enabled=bool(g("clutter", "enabled", False)),
            clutter_contact_ok=bool(g("clutter", "contact_ok", False)),
        )

    @property
    def _default_sensor_configs(self):
        scene_pose = sapien_utils.look_at(eye=[0.5, 0.0, 0.7], target=[0.0, 0.0, 0.05])
        configs = [
            CameraConfig(
                "scene_camera",
                scene_pose,
                self.camera_width,
                self.camera_height,
                1.0,
                0.01,
                100,
            ),
        ]
        # Wrist-mounted view (Panda URDF includes camera_link; agent is loaded before sensors).
        if self.agent is not None and "camera_link" in self.agent.robot.links_map:
            configs.append(
                CameraConfig(
                    "arm_camera",
                    sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                    self.camera_width,
                    self.camera_height,
                    fov=np.pi / 2,
                    near=0.01,
                    far=100,
                    mount=self.agent.robot.links_map["camera_link"],
                )
            )
        return configs

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.5, 0.0, 0.7], target=[0.0, 0.0, 0.05])
        return CameraConfig("render_camera", pose, 512, 512, 1.0, 0.01, 100)

    def render(self):
        """render_mode='all': side-by-side [render | sensor] instead of ManiSkill's padded tiling."""
        if self.render_mode == "all":
            render = self.render_rgb_array()
            sensor = self.get_sensor_images()["scene_camera"]["rgb"]
            h = render.shape[1]
            w = torch.nn.functional.interpolate(
                sensor.permute(0, 3, 1, 2).float(), size=(h, h), mode="nearest"
            ).permute(0, 2, 3, 1).to(render.dtype)
            return torch.cat([render, w], dim=2)
        return super().render()

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_lighting(self, options: dict):
        rng = self._batched_episode_rng
        lo, hi = self._rand["light_intensity"]
        inten = float(rng[0].uniform(lo, hi)) if hi > lo else float(lo)
        dlo, dhi = self._rand["light_dir_jitter"]
        dj = float(rng[0].uniform(dlo, dhi)) if dhi > dlo else 0.0
        self.scene.set_ambient_light([0.3 * inten, 0.3 * inten, 0.3 * inten])
        self.scene.add_directional_light(
            [1 + dj, 1 + dj, -1], [inten, inten, inten],
            shadow=self.enable_shadow, shadow_scale=5, shadow_map_size=2048,
        )
        self.scene.add_directional_light([0, 0, -1], [0.5 * inten] * 3)

    def _build_bin(self, color, name: str):
        bx, by = self.bin_half
        h, t, ft = self.bin_wall_h, self.bin_wall_t, self.bin_floor_t
        builder = self.scene.create_actor_builder()
        mat = sapien.render.RenderMaterial(base_color=[*color, 1.0])
        builder.add_box_collision(pose=sapien.Pose(p=[0, 0, ft]), half_size=[bx, by, ft])
        builder.add_box_visual(pose=sapien.Pose(p=[0, 0, ft]), half_size=[bx, by, ft], material=mat)
        walls = [
            ([bx, 0, h], [t, by, h]),
            ([-bx, 0, h], [t, by, h]),
            ([0, by, h], [bx, t, h]),
            ([0, -by, h], [bx, t, h]),
        ]
        for p, hs in walls:
            builder.add_box_collision(pose=sapien.Pose(p=p), half_size=hs)
            builder.add_box_visual(pose=sapien.Pose(p=p), half_size=hs, material=mat)
        builder.initial_pose = sapien.Pose(p=[0, 0, 0])
        return builder.build_kinematic(name=name)

    def _build_parcel(self, idx: int, tag_id: int):
        phx, phy, phz = self.parcel_half
        thx, thy, thz = self.tag_half
        rng = self._batched_episode_rng
        per_env = []
        for i in range(self.num_envs):
            cs_lo, cs_hi = self._rand["cardboard_shade"]
            ts_lo, ts_hi = self._rand["tag_shade"]
            c_off = float(rng[i].uniform(cs_lo, cs_hi)) if cs_hi > cs_lo else 0.0
            t_off = float(rng[i].uniform(ts_lo, ts_hi)) if ts_hi > ts_lo else 0.0
            cardboard = np.clip(np.array(CARDBOARD_BASE) + c_off, 0.05, 0.95)
            tagcol = np.clip(np.array(TAG_BASE_COLORS[tag_id]) + t_off, 0.05, 0.98)
            b = self.scene.create_actor_builder()
            b.add_box_collision(half_size=[phx, phy, phz])
            b.add_box_visual(
                half_size=[phx, phy, phz],
                material=sapien.render.RenderMaterial(base_color=[*cardboard.tolist(), 1.0]),
            )
            tag_x = phx - thx - 0.004
            tag_y = phy - thy - 0.004
            b.add_box_visual(
                pose=sapien.Pose(p=[-tag_x, tag_y, phz + thz]),
                half_size=[thx, thy, thz],
                material=sapien.render.RenderMaterial(base_color=[*tagcol.tolist(), 1.0]),
            )
            b.set_scene_idxs([i])
            b.initial_pose = sapien.Pose(p=[0, 0, phz + 0.5 * i])  # spread to avoid init overlap
            per_env.append(b.build_dynamic(name=f"parcel_{idx}_env{i}"))
        return Actor.merge(per_env, name=f"parcel_{idx}")

    def _build_background(self):
        rng = self._batched_episode_rng
        table_colors = self._rand["table_colors"]
        floor_colors = self._rand["floor_colors"]
        tops, floors = [], []
        for i in range(self.num_envs):
            tc = table_colors[int(rng[i].randint(0, len(table_colors)))]
            fc = floor_colors[int(rng[i].randint(0, len(floor_colors)))]
            tb = self.scene.create_actor_builder()
            tb.add_box_visual(half_size=[0.66, 1.25, 0.0015],
                              material=sapien.render.RenderMaterial(base_color=[*tc, 1.0]))
            tb.set_scene_idxs([i])
            tb.initial_pose = sapien.Pose(p=[-0.135, 0, 0.0015])
            tops.append(tb.build_static(name=f"table_surface_env{i}"))
            fb = self.scene.create_actor_builder()
            fb.add_box_visual(half_size=[2.5, 2.5, 0.001],
                              material=sapien.render.RenderMaterial(base_color=[*fc, 1.0]))
            fb.set_scene_idxs([i])
            fb.initial_pose = sapien.Pose(p=[0, 0, -0.5])
            floors.append(fb.build_static(name=f"floor_mat_env{i}"))
        self.table_surface = Actor.merge(tops, name="table_surface")
        self.floor_mat = Actor.merge(floors, name="floor_mat")

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self._build_background()
        self.bins = [
            self._build_bin(BIN_BASE_COLORS[0], "bin_red"),
            self._build_bin(BIN_BASE_COLORS[1], "bin_blue"),
        ]
        self.parcels = [self._build_parcel(j, t) for j, t in enumerate(self._parcel_tag_ids)]
        self.parcel_tags = torch.tensor(self._parcel_tag_ids, device=self.device).long()
        self.parcel_tags = self.parcel_tags[None].repeat(self.num_envs, 1)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            qpos = torch.tensor(self.START_QPOS, device=self.device)
            self.agent.reset(qpos.unsqueeze(0).repeat(b, 1))
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

            swap_p = self._rand["bin_side_swap_prob"]
            swap = torch.rand(b) < swap_p
            jx_lo, jx_hi = self._rand["bin_xy_jitter"]
            for color_id, bin_actor in enumerate(self.bins):
                sign = torch.where(swap, torch.tensor(-1.0), torch.tensor(1.0))
                if color_id == 1:
                    sign = -sign
                pos = torch.zeros((b, 3))
                pos[:, 0] = self.bin_base_x
                pos[:, 1] = sign * self.bin_base_y
                if jx_hi > jx_lo:
                    pos[:, 0] += torch.rand(b) * (jx_hi - jx_lo) + jx_lo
                    pos[:, 1] += torch.rand(b) * (jx_hi - jx_lo) + jx_lo
                bin_actor.set_pose(Pose.create_from_pq(pos))

            n = self.num_parcels
            cx, cy = self.inbound_center
            jx_lo, jx_hi = self._rand["parcel_xy_jitter"]
            yaw_lo, yaw_hi = self._rand["parcel_yaw_jitter"]
            cols = min(2, n)
            rows = int(np.ceil(n / cols))
            sx = 2 * self.parcel_half[0] + 0.06
            sy = 2 * self.parcel_half[1] + 0.06
            for j, parcel in enumerate(self.parcels):
                r, c = divmod(j, cols)
                gx = cx + (c - (cols - 1) / 2.0) * sx
                gy = cy + (r - (rows - 1) / 2.0) * sy
                pos = torch.zeros((b, 3))
                pos[:, 0] = gx
                pos[:, 1] = gy
                pos[:, 2] = self.parcel_half[2] + 0.001
                if not self.fixed_poses and jx_hi > jx_lo:
                    pos[:, 0] += torch.rand(b) * (jx_hi - jx_lo) + jx_lo
                    pos[:, 1] += torch.rand(b) * (jx_hi - jx_lo) + jx_lo
                if not self.fixed_poses and yaw_hi > yaw_lo:
                    yaw = torch.rand(b) * (yaw_hi - yaw_lo) + yaw_lo
                else:
                    yaw = torch.zeros(b)
                quat = torch.zeros((b, 4))
                quat[:, 0] = torch.cos(yaw / 2)
                quat[:, 3] = torch.sin(yaw / 2)
                parcel.set_pose(Pose.create_from_pq(pos, quat))

            N, P = self.num_envs, self.num_parcels
            if not hasattr(self, "_prev_sorted") or self._prev_sorted.shape[0] != N:
                self._prev_sorted = torch.zeros(N, device=self.device)
                self._steps_to_complete = torch.full(
                    (N,), self.max_episode_steps, dtype=torch.long, device=self.device
                )
                self._placed_correct = torch.zeros(N, P, dtype=torch.bool, device=self.device)
                self._placed_other = torch.zeros(N, P, dtype=torch.bool, device=self.device)
            self._prev_sorted[env_idx] = 0.0
            self._steps_to_complete[env_idx] = self.max_episode_steps
            self._placed_correct[env_idx] = False
            self._placed_other[env_idx] = False

    def _bin_positions(self):
        """(num_envs, 2, 3): xyz of bin 0 (red) then bin 1 (blue)."""
        return torch.stack([b.pose.p for b in self.bins], dim=1)

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,
            is_grasped=info["is_grasped"],
        )
        if "state" in self.obs_mode:
            parcel_pose = torch.stack([p.pose.raw_pose for p in self.parcels], dim=1)
            tag_onehot = torch.nn.functional.one_hot(self.parcel_tags, num_classes=2).float()
            bin_pos = self._bin_positions()
            n_bins = len(self.bins)
            bin_color_onehot = torch.eye(n_bins, device=self.device)[None].repeat(self.num_envs, 1, 1)
            obs.update(
                parcel_pose=parcel_pose.reshape(self.num_envs, -1),
                parcel_tag=tag_onehot.reshape(self.num_envs, -1),
                bin_position=bin_pos.reshape(self.num_envs, -1),
                bin_color=bin_color_onehot.reshape(self.num_envs, -1),
            )
        return obs

    def evaluate(self):
        grasp = torch.stack(
            [self.agent.is_grasping(p) for p in self.parcels], dim=1
        )
        is_grasped = grasp.any(dim=1)

        bin_pos = self._bin_positions()
        bx, by = self.bin_half
        rim_z = 0.06

        for j, parcel in enumerate(self.parcels):
            p = parcel.pose.p
            tag = self.parcel_tags[:, j]
            correct_bin = bin_pos[torch.arange(self.num_envs), tag]

            def inside(b):
                return (
                    (torch.abs(p[:, 0] - b[:, 0]) < bx)
                    & (torch.abs(p[:, 1] - b[:, 1]) < by)
                    & (p[:, 2] < rim_z)
                    & (p[:, 2] > 0.0)
                )

            released = ~grasp[:, j]
            self._placed_correct[:, j] |= inside(correct_bin) & released
            other_bin = bin_pos[torch.arange(self.num_envs), 1 - tag]
            self._placed_other[:, j] |= inside(other_bin) & released & ~self._placed_correct[:, j]

        correct = self._placed_correct.float().sum(dim=1)
        mis = self._placed_other.float().sum(dim=1)
        placed = (self._placed_correct | self._placed_other).float().sum(dim=1)

        all_placed = placed >= self.num_parcels
        newly_done = all_placed & (self._steps_to_complete == self.max_episode_steps)
        self._steps_to_complete = torch.where(
            newly_done, self.elapsed_steps.long(), self._steps_to_complete
        )

        return {
            "success_count": correct,
            "sort_accuracy": correct / self.num_parcels,
            "mis_sort_count": mis,
            "all_placed": all_placed,
            "steps_to_complete": self._steps_to_complete,
            "is_grasped": is_grasped,
            "success": all_placed,
        }

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        cur = info["success_count"]
        delta = torch.clamp(cur - self._prev_sorted, min=0.0)
        self._prev_sorted = cur
        return delta

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        raise NotImplementedError(
            "No dense reward provided. Implement compute_dense_reward() yourself, "
            "or use reward_mode='sparse' (the default)."
        )

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        raise NotImplementedError("No dense reward provided.")
