ALGO_NAME = "BC_Diffusion_rgbd_UNet"

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import gymnasium as gym
from gymnasium.vector.vector_env import VectorEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from tqdm import tqdm
import tyro
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from gymnasium import spaces
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.utils import (IterationBasedBatchSampler,
                                    build_state_obs_extractor, convert_obs,
                                    worker_init_fn)


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    skip_env_eval: bool = False
    """skip creating sim eval envs (train on demos only — for hosts without Vulkan)"""

    env_id: str = "PegInsertionSide-v1"
    """the id of the environment"""
    demo_path: str = (
        "demos/PegInsertionSide-v1/trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
    )
    """the path of demo dataset, it is expected to be a ManiSkill dataset h5py format file"""
    extra_demo_paths: Optional[str] = None
    """comma-separated paths to ADDITIONAL demo .h5 files to train on jointly (e.g. medium+hard).
    Combined with --demo-path so one policy learns easy+medium+hard at once."""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset (applied per demo file)"""

    # ----- Data augmentation (image IL robustness + free extra data) ---------------------
    visual_aug: bool = False
    """apply colour-jitter + gaussian-blur to the RGB obs during training (keeps colour cues,
    just harder). Helps the policy generalise to the held-out eval's lighting/appearance jitter."""
    mirror_prob: float = 0.0
    """probability of applying a horizontal MIRROR to a training sample: flip the scene-cam image
    left<->right, negate the world-y action delta, and mirror the robot proprioception. The scene
    camera looks down -x with +z up, so image-horizontal == world-y; bins live at +/-y and 'hard'
    swaps their sides, so mirroring teaches colour-routing symmetry and doubles the data. 0 disables."""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""

    # Diffusion Policy specific arguments
    lr: float = 1e-4
    """the learning rate of the diffusion policy"""
    obs_horizon: int = 2  # Seems not very important in ManiSkill, 1, 2, 4 work well
    act_horizon: int = 8  # Seems not very important in ManiSkill, 4, 8, 15 work well
    pred_horizon: int = (
        16  # 16->8 leads to worse performance, maybe it is like generate a half image; 16->32, improvement is very marginal
    )
    diffusion_step_embed_dim: int = 64  # not very important
    unet_dims: List[int] = field(
        default_factory=lambda: [64, 128, 256]
    )  # default setting is about ~4.5M params
    n_groups: int = (
        8  # jigu says it is better to let each group have at least 8 channels; it seems 4 and 8 are simila
    )

    # Environment/experiment specific arguments
    obs_mode: str = "rgb+depth"
    """The observation mode to use for the environment, which dictates what visual inputs to pass to the model. Can be "rgb", "depth", or "rgb+depth"."""
    obs_camera: str = "scene"
    """Accepted for demo-replay compat; WarehouseSort only uses the fixed third-person scene camera."""
    visual_encoder: str = "plain_conv"
    """RGB encoder: "plain_conv" (vendored) or "resnet18" (ResNet18 + SpatialSoftmax keypoints)."""
    num_kp: int = 32
    """SpatialSoftmax keypoints (resnet18 encoder); 2*num_kp coords localise parcels+bins+gripper."""
    # WarehouseSort scene knobs (so the eval env matches the demos). Defaults = easy.
    num_parcels: int = 2
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    eval_freq: int = 5000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = None
    """the frequency of saving numbered checkpoints (also updates latest.pt). None = eval-best only (or best_train_loss when --skip-env-eval)."""
    resume_from: Optional[str] = None
    """path to a checkpoint (.pt) to resume training from (loads weights, optimizer, scheduler, EMA, iteration)."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "physx_cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None


def reorder_keys(d, ref_dict):
    out = dict()
    for k, v in ref_dict.items():
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out


class SmallDemoDataset_DiffusionPolicy(Dataset):  # Load everything into memory
    def __init__(self, data_path, obs_process_fn, obs_space, include_rgb, include_depth,
                 device, num_traj, visual_aug=False, mirror_prob=0.0):
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        self.device = device
        from diffusion_policy.utils import load_demo_dataset

        # Accept a single path or a list of paths -> train on the union of all demo files
        # (e.g. easy + medium + hard) so a single policy generalises across difficulties.
        if isinstance(data_path, str):
            data_paths = [data_path]
        else:
            data_paths = list(data_path)

        all_obs, all_actions = [], []
        for dp in data_paths:
            print(f"Loading demos from: {dp}")
            traj = load_demo_dataset(dp, num_traj=num_traj, concat=False)
            # traj['observations'] is a list of dict (one per traj), values length L+1
            # traj['actions'] is a list of np.ndarray (L, act_dim)
            for obs_traj_dict in traj["observations"]:
                _obs_traj_dict = reorder_keys(
                    obs_traj_dict, obs_space
                )  # key order in demo is different from key order in env obs
                _obs_traj_dict = obs_process_fn(_obs_traj_dict)
                if self.include_depth:
                    _obs_traj_dict["depth"] = torch.Tensor(
                        _obs_traj_dict["depth"].astype(np.float32)
                    ).to(device=device, dtype=torch.float16)
                if self.include_rgb:
                    _obs_traj_dict["rgb"] = torch.from_numpy(_obs_traj_dict["rgb"]).to(
                        device
                    )  # still uint8
                _obs_traj_dict["state"] = torch.from_numpy(_obs_traj_dict["state"]).to(
                    device
                )
                all_obs.append(_obs_traj_dict)
            for a in traj["actions"]:
                all_actions.append(torch.Tensor(a).to(device=device))

        trajectories = {"observations": all_obs, "actions": all_actions}
        self.obs_keys = list(all_obs[0].keys())
        print(
            f"Loaded {len(all_actions)} trajectories from {len(data_paths)} demo file(s). "
            "Obs/action pre-processing is done, start to pre-compute the slice indices..."
        )

        # --- augmentation setup ---------------------------------------------------------
        # Visual augmentation lives in the Dataset (per the challenge guidance): keep hue
        # adjustments tiny so the red/blue colour-coding the task depends on survives.
        self.visual_augmentations = (
            T.Compose([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
                T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5)),
            ])
            if visual_aug
            else None
        )
        self.mirror_prob = float(mirror_prob)
        # State layout = [qpos(9: 7 arm + 2 gripper), qvel(9), tcp_pose(7: xyz + quat wxyz),
        # is_grasped(1)] = 26. A horizontal flip mirrors the world about y=0: negate the
        # z-axis arm joints (Franka J1/J3/J5/J7 -> idx 0,2,4,6) for qpos & qvel, negate the
        # tcp y-position and the quaternion's x,z parts. is_grasped & gripper width unchanged.
        self._mirror_state_idx = None
        if all_obs[0]["state"].shape[-1] == 26:
            self._mirror_state_idx = torch.tensor(
                [0, 2, 4, 6, 9, 11, 13, 15, 19, 22, 24], device=device, dtype=torch.long
            )
        elif self.mirror_prob > 0:
            print(
                f"[mirror] state dim is {all_obs[0]['state'].shape[-1]} (expected 26); "
                "mirroring image+action only, proprioception left unmirrored."
            )

        # Pre-compute all possible (traj_idx, start, end) tuples, this is very specific to Diffusion Policy
        if (
            "delta_pos" in args.control_mode
            or args.control_mode == "base_pd_joint_vel_arm_pd_joint_vel"
        ):
            print("Detected a delta controller type, padding with a zero action to ensure the arm stays still after solving tasks.")
            self.pad_action_arm = torch.zeros(
                (trajectories["actions"][0].shape[1] - 1,), device=device
            )
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
        else:
            # NOTE for absolute joint pos control probably should pad with the final joint position action.
            raise NotImplementedError(f"Control Mode {args.control_mode} not supported")
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon = (
            args.obs_horizon,
            args.pred_horizon,
        )
        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        for traj_idx in range(num_traj):
            L = trajectories["actions"][traj_idx].shape[0]
            assert trajectories["observations"][traj_idx]["state"].shape[0] == L + 1
            total_transitions += L

            # |o|o|                             observations: 2
            # | |a|a|a|a|a|a|a|a|               actions executed: 8
            # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
            pad_before = obs_horizon - 1
            # Pad before the trajectory, so the first action of an episode is in "actions executed"
            # obs_horizon - 1 is the number of "not used actions"
            pad_after = pred_horizon - obs_horizon
            # Pad after the trajectory, so all the observations are utilized in training
            # Note that in the original code, pad_after = act_horizon - 1, but I think this is not the best choice
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]  # slice indices follow convention [start, end)

        print(
            f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}"
        )

        self.trajectories = trajectories

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape

        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[
                max(0, start) : start + self.obs_horizon
            ]  # start+self.obs_horizon is at least 1
            if start < 0:  # pad before the trajectory
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)
            # don't need to pad obs after the trajectory, see the above char drawing

        act_seq = self.trajectories["actions"][traj_idx][max(0, start) : end]
        if start < 0:  # pad before the trajectory
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:  # pad after the trajectory
            gripper_action = act_seq[-1, -1]  # assume gripper is with pos controller
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
            # making the robot (arm and gripper) stay still
        assert (
            obs_seq["state"].shape[0] == self.obs_horizon
            and act_seq.shape[0] == self.pred_horizon
        )

        # ----- augmentation (applied on the sampled window) ---------------------------
        if self.visual_augmentations is not None and self.include_rgb:
            # obs_seq['rgb']: (obs_horizon, C, H, W) uint8; same jitter/blur across the stack.
            obs_seq["rgb"] = self.visual_augmentations(obs_seq["rgb"])
        if self.mirror_prob > 0.0 and random.random() < self.mirror_prob:
            obs_seq, act_seq = self._mirror(obs_seq, act_seq)

        return {
            "observations": obs_seq,
            "actions": act_seq,
        }

    def _mirror(self, obs_seq, act_seq):
        """Horizontal mirror about the world y=0 plane (image-horizontal == world-y).

        Flip the scene image left<->right, negate the world-y action delta (action idx 1),
        and mirror the robot proprioception so image and state stay consistent.
        """
        obs_seq = dict(obs_seq)
        if self.include_rgb and "rgb" in obs_seq:
            obs_seq["rgb"] = torch.flip(obs_seq["rgb"], dims=[-1])  # flip width (new tensor)
        if self.include_depth and "depth" in obs_seq:
            obs_seq["depth"] = torch.flip(obs_seq["depth"], dims=[-1])
        state = obs_seq["state"].clone()
        if self._mirror_state_idx is not None:
            state[:, self._mirror_state_idx] = -state[:, self._mirror_state_idx]
        obs_seq["state"] = state
        act = act_seq.clone()
        act[:, 1] = -act[:, 1]  # negate world-y end-effector delta
        return obs_seq, act

    def __len__(self):
        return len(self.slices)


class Agent(nn.Module):
    def __init__(self, env: VectorEnv, args: Args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        assert (
            len(env.single_observation_space["state"].shape) == 2
        )  # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1  # (act_dim, )
        assert (env.single_action_space.high == 1).all() and (
            env.single_action_space.low == -1
        ).all()
        # denoising results will be clipped to [-1,1], so the action should be in [-1,1] as well
        self.act_dim = env.single_action_space.shape[0]
        obs_state_dim = env.single_observation_space["state"].shape[1]
        total_visual_channels = 0
        self.include_rgb = "rgb" in env.single_observation_space.keys()
        self.include_depth = "depth" in env.single_observation_space.keys()

        if self.include_rgb:
            total_visual_channels += env.single_observation_space["rgb"].shape[-1]
        if self.include_depth:
            total_visual_channels += env.single_observation_space["depth"].shape[-1]

        visual_feature_dim = 256
        enc = getattr(args, "visual_encoder", "plain_conv")
        if enc in ("convnext", "convnext_tiny"):
            # ConvNeXt-Tiny + SpatialSoftmax: stronger, modern perception backbone (LayerNorm ->
            # batch-size independent) with the same keypoint head. Best generalisation here.
            from diffusion_policy.lerobot_encoder import ConvNeXtSpatialSoftmax
            self.visual_encoder = ConvNeXtSpatialSoftmax(
                in_channels=total_visual_channels, out_dim=visual_feature_dim,
                num_kp=getattr(args, "num_kp", 32),
            )
        elif enc == "resnet18":
            # ResNet18 + SpatialSoftmax: encodes object/gripper LOCATIONS as keypoint coords
            # (see lerobot_encoder). Best for spatial pick-and-place.
            from diffusion_policy.lerobot_encoder import ResNet18SpatialSoftmax
            self.visual_encoder = ResNet18SpatialSoftmax(
                in_channels=total_visual_channels, out_dim=visual_feature_dim,
                num_kp=getattr(args, "num_kp", 32),
            )
        else:
            # pool_feature_map=False: flatten the full 8x8 conv feature map instead of global
            # max-pooling it to 1x1 (which discards WHERE objects are and makes the policy collapse).
            self.visual_encoder = PlainConv(
                in_channels=total_visual_channels, out_dim=visual_feature_dim, pool_feature_map=False
            )
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,  # act_horizon is not used (U-Net doesn't care)
            global_cond_dim=self.obs_horizon * (visual_feature_dim + obs_state_dim),
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",  # has big impact on performance, try not to change
            clip_sample=True,  # clip output to [-1,1] to improve stability
            prediction_type="epsilon",  # predict noise (instead of denoised action)
        )

    def encode_obs(self, obs_seq, eval_mode):
        if self.include_rgb:
            rgb = obs_seq["rgb"].float() / 255.0  # (B, obs_horizon, 3*k, H, W)
            img_seq = rgb
        if self.include_depth:
            depth = obs_seq["depth"].float() / 1024.0  # (B, obs_horizon, 1*k, H, W)
            img_seq = depth
        if self.include_rgb and self.include_depth:
            img_seq = torch.cat([rgb, depth], dim=2)  # (B, obs_horizon, C, H, W), C=4*k
        batch_size = img_seq.shape[0]
        img_seq = img_seq.flatten(end_dim=1)  # (B*obs_horizon, C, H, W)
        if hasattr(self, "aug") and not eval_mode:
            img_seq = self.aug(img_seq)  # (B*obs_horizon, C, H, W)
        visual_feature = self.visual_encoder(img_seq)  # (B*obs_horizon, D)
        visual_feature = visual_feature.reshape(
            batch_size, self.obs_horizon, visual_feature.shape[1]
        )  # (B, obs_horizon, D)
        feature = torch.cat(
            (visual_feature, obs_seq["state"]), dim=-1
        )  # (B, obs_horizon, D+obs_state_dim)
        return feature.flatten(start_dim=1)  # (B, obs_horizon * (D+obs_state_dim))

    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq["state"].shape[0]

        # observation as FiLM conditioning
        obs_cond = self.encode_obs(
            obs_seq, eval_mode=False
        )  # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device
        ).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(
            noisy_action_seq, timesteps, global_cond=obs_cond
        )

        return F.mse_loss(noise_pred, noise)

    def get_action(self, obs_seq):
        # init scheduler
        # self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        # set_timesteps will change noise_scheduler.timesteps is only used in noise_scheduler.step()
        # noise_scheduler.step() is only called during inference
        # if we use DDPM, and inference_diffusion_steps == train_diffusion_steps, then we can skip this

        # obs_seq['state']: (B, obs_horizon, obs_state_dim)
        B = obs_seq["state"].shape[0]
        with torch.no_grad():
            if self.include_rgb:
                obs_seq["rgb"] = obs_seq["rgb"].permute(0, 1, 4, 2, 3)
            if self.include_depth:
                obs_seq["depth"] = obs_seq["depth"].permute(0, 1, 4, 2, 3)

            obs_cond = self.encode_obs(
                obs_seq, eval_mode=True
            )  # (B, obs_horizon * obs_dim)

            # initialize action from Guassian noise
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=obs_seq["state"].device
            )

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )

                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]  # (B, act_horizon, act_dim)


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    if args.demo_path.endswith(".h5"):
        import json

        json_file = args.demo_path[:-2] + "json"
        with open(json_file, "r") as f:
            demo_info = json.load(f)
            if "control_mode" in demo_info["env_info"]["env_kwargs"]:
                control_mode = demo_info["env_info"]["env_kwargs"]["control_mode"]
            elif "control_mode" in demo_info["episodes"][0]:
                control_mode = demo_info["episodes"][0]["control_mode"]
            else:
                raise Exception("Control mode not found in json")
            assert (
                control_mode == args.control_mode
            ), f"Control mode mismatched. Dataset has control mode {control_mode}, but args has control mode {args.control_mode}"
    # Match the eval env to the demo distribution: pull the WarehouseSort scene kwargs straight
    # from the demo's recorded env_kwargs (num parcels, bins, distance, pose randomisation, camera)
    # so eval renders exactly what the policy was trained on (no manual flag duplication).
    _demo_scene_kwargs = {}
    if args.demo_path.endswith(".h5") and args.env_id.startswith("WarehouseSort"):
        _dk = demo_info["env_info"]["env_kwargs"]
        for _k in ("num_parcels", "fixed_poses", "randomization", "obs_camera"):
            if _k in _dk:
                _demo_scene_kwargs[_k] = _dk[_k]
    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
    assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # create evaluation environment (optional — not needed for demo-only training)
    envs = None
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        obs_camera=args.obs_camera,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default")
    )
    if args.env_id.startswith("WarehouseSort"):   # match the demo scene (num parcels, poses, rand)
        env_kwargs.update(num_parcels=args.num_parcels)
        env_kwargs.update(_demo_scene_kwargs)      # demo-recorded kwargs win (exact distribution match)
    assert args.max_episode_steps != None, "max_episode_steps must be specified as imitation learning algorithms task solve speed is dependent on the data you train on"
    env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    if not args.skip_env_eval:
        envs = make_eval_envs(
            args.env_id,
            args.num_eval_envs,
            args.sim_backend,
            env_kwargs,
            other_kwargs,
            video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
            wrappers=[FlattenRGBDObservationWrapper],
        )
    elif args.skip_env_eval:
        print("skip_env_eval=True — training on demos only (no sim eval / videos).")

    if args.track:
        import wandb
        config = vars(args)
        config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, env_horizon=args.max_episode_steps)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="DiffusionPolicy",
            tags=["diffusion_policy"],
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    obs_process_fn = partial(
        convert_obs,
        concat_fn=partial(np.concatenate, axis=-1),
        transpose_fn=partial(
            np.transpose, axes=(0, 3, 1, 2)
        ),  # (B, H, W, C) -> (B, C, H, W)
        state_obs_extractor=build_state_obs_extractor(args.env_id),
        depth = "rgbd" in args.demo_path
    )

    # create temporary env to get original observation space as AsyncVectorEnv (CPU parallelization) doesn't permit that
    # (use the SAME sim backend as the eval envs so this throwaway env doesn't spin up a CPU PhysX
    # system; eval itself runs on args.sim_backend, i.e. GPU here)
    if args.skip_env_eval:
        # No sim available (e.g. no Vulkan on headless host).
        # Build nested obs space from the h5 file structure so reorder_keys works correctly.
        import h5py

        def _space_from_h5_group(grp):
            if isinstance(grp, h5py.Dataset):
                shape = grp.shape[1:]  # drop time axis
                if grp.dtype == bool or str(grp.dtype) == 'bool':
                    return spaces.Box(0, 1, shape if shape else (1,), np.int8)
                elif np.issubdtype(grp.dtype, np.floating):
                    return spaces.Box(-np.inf, np.inf, shape if shape else (1,), np.float32)
                else:
                    return spaces.Box(0, 255, shape if shape else (1,), grp.dtype)
            else:
                return spaces.Dict({k: _space_from_h5_group(grp[k]) for k in grp.keys()})

        with h5py.File(args.demo_path, "r") as _f:
            _obs_grp = _f["traj_0"]["obs"]
            orignal_obs_space = _space_from_h5_group(_obs_grp)
            _act_dim = int(_f["traj_0"]["actions"].shape[-1])
            # state = agent (qpos, qvel) + extra (tcp_pose, is_grasped) — matches build_state_obs_extractor
            _state_dim = sum(
                int(ds.shape[-1]) if len(ds.shape) > 1 else 1
                for grp_name in ["agent", "extra"]
                for ds in _obs_grp[grp_name].values()
            )
            _rgb_h  = int(_obs_grp["sensor_data"]["scene_camera"]["rgb"].shape[1])
            _rgb_w  = int(_obs_grp["sensor_data"]["scene_camera"]["rgb"].shape[2])
            _rgb_c  = int(_obs_grp["sensor_data"]["scene_camera"]["rgb"].shape[3])

        include_rgb   = True
        include_depth = False

        class _FakeEnv:
            single_observation_space = spaces.Dict({
                "rgb":   spaces.Box(0, 255, (args.obs_horizon, _rgb_h, _rgb_w, _rgb_c), np.uint8),
                "state": spaces.Box(-np.inf, np.inf, (args.obs_horizon, _state_dim), np.float32),
            })
            single_action_space = spaces.Box(
                np.full(_act_dim, -1, np.float32),
                np.full(_act_dim,  1, np.float32),
                (_act_dim,), np.float32,
            )

            def close(self):
                pass

        envs = _FakeEnv()
    else:
        tmp_env = gym.make(args.env_id, sim_backend=args.sim_backend, **env_kwargs)
        orignal_obs_space = tmp_env.observation_space
        include_rgb   = tmp_env.unwrapped.obs_mode_struct.visual.rgb
        include_depth = tmp_env.unwrapped.obs_mode_struct.visual.depth
        tmp_env.close()

    # Combine the primary demo with any extra demo files (e.g. medium + hard) so a single
    # policy is trained jointly across difficulties.
    all_demo_paths = [args.demo_path]
    if args.extra_demo_paths:
        all_demo_paths += [p for p in args.extra_demo_paths.split(",") if p.strip()]
    if len(all_demo_paths) > 1:
        print(f"Training jointly on {len(all_demo_paths)} demo files: {all_demo_paths}")

    dataset = SmallDemoDataset_DiffusionPolicy(
        data_path=all_demo_paths,
        obs_process_fn=obs_process_fn,
        obs_space=orignal_obs_space,
        include_rgb=include_rgb,
        include_depth=include_depth,
        device=device,
        num_traj=args.num_demos,
        visual_aug=args.visual_aug,
        mirror_prob=args.mirror_prob,
    )
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    start_iteration = 0
    train_state = {"best_train_loss": float("inf")}
    resume_path = None
    if args.resume_from:
        resume_path = args.resume_from
        if not os.path.isabs(resume_path):
            resume_path = os.path.join(os.getcwd(), resume_path)
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"Resuming from checkpoint: {resume_path}")
    batch_sampler = IterationBasedBatchSampler(
        batch_sampler, args.total_iters, start_iter=start_iteration
    )
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
        persistent_workers=(args.num_dataload_workers > 0),
    )

    agent = Agent(envs, args).to(device)

    optimizer = optim.AdamW(
        params=agent.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6
    )

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(envs, args).to(device)

    if args.resume_from:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["agent"])
        ema_agent.load_state_dict(ckpt.get("ema_agent", ckpt["agent"]))
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_iteration = ckpt.get("iteration", -1) + 1
        train_state["best_train_loss"] = ckpt.get(
            "best_train_loss", train_state["best_train_loss"]
        )
        best_eval_metrics = defaultdict(float, ckpt.get("best_eval_metrics", {}))
        batch_sampler.start_iter = start_iteration
        print(
            f"Resumed at iteration {start_iteration} "
            f"(best train loss={train_state['best_train_loss']:.4f})"
        )
    else:
        best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    ckpt_dir = f"runs/{run_name}/checkpoints"

    def save_ckpt(tag, iteration, *, update_latest=True):
        os.makedirs(ckpt_dir, exist_ok=True)
        ema.copy_to(ema_agent.parameters())
        payload = {
            "agent": agent.state_dict(),
            "ema_agent": ema_agent.state_dict(),
            "iteration": iteration,
            "best_train_loss": train_state["best_train_loss"],
            "best_eval_metrics": dict(best_eval_metrics),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "ema": ema.state_dict(),
            "args": vars(args),
        }
        path = f"{ckpt_dir}/{tag}.pt"
        torch.save(payload, path)
        if update_latest:
            torch.save(payload, f"{ckpt_dir}/latest.pt")
        print(f"Saved checkpoint: {path}")

    def maybe_save_periodic(iteration):
        if args.save_freq is None or iteration <= 0 or iteration % args.save_freq != 0:
            return
        save_ckpt(str(iteration), iteration)

    def maybe_save_best_train_loss(iteration, loss_value):
        if not args.skip_env_eval or iteration <= 0:
            return
        if loss_value >= train_state["best_train_loss"]:
            return
        train_state["best_train_loss"] = loss_value
        save_ckpt("best_train_loss", iteration, update_latest=False)
        print(f"New best train loss: {loss_value:.4f}")

    # define evaluation and logging functions
    def evaluate_and_save_best(iteration):
        if args.skip_env_eval:
            return
        if iteration % args.eval_freq == 0:
            last_tick = time.time()
            ema.copy_to(ema_agent.parameters())
            eval_metrics = evaluate(
                args.num_eval_episodes, ema_agent, envs, device, args.sim_backend
            )
            timings["eval"] += time.time() - last_tick

            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                print(f"{k}: {eval_metrics[k]:.4f}")

            save_on_best_metrics = ["sort_accuracy", "success_once", "success_at_end"]
            for k in save_on_best_metrics:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    save_ckpt(f"best_eval_{k}", iteration, update_latest=False)
                    print(
                        f"New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint."
                    )
    def log_metrics(iteration):
        if iteration % args.log_freq == 0:
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], iteration
            )
            writer.add_scalar("losses/total_loss", total_loss.item(), iteration)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, iteration)

    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- #
    agent.train()
    pbar = tqdm(total=args.total_iters, initial=start_iteration)
    last_tick = time.time()
    for iteration, data_batch in enumerate(train_dataloader, start=start_iteration):
        timings["data_loading"] += time.time() - last_tick

        # forward and compute loss
        last_tick = time.time()
        total_loss = agent.compute_loss(
            obs_seq=data_batch["observations"],  # obs_batch_dict['state'] is (B, L, obs_dim)
            action_seq=data_batch["actions"],  # (B, L, act_dim)
        )
        timings["forward"] += time.time() - last_tick

        # backward
        last_tick = time.time()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()  # step lr scheduler every batch, this is different from standard pytorch behavior
        timings["backward"] += time.time() - last_tick

        # ema step
        last_tick = time.time()
        ema.step(agent.parameters())
        timings["ema"] += time.time() - last_tick

        evaluate_and_save_best(iteration)
        log_metrics(iteration)
        maybe_save_periodic(iteration)
        maybe_save_best_train_loss(iteration, total_loss.item())
        pbar.update(1)
        pbar.set_postfix({"loss": total_loss.item()})
        last_tick = time.time()

    evaluate_and_save_best(args.total_iters)
    log_metrics(args.total_iters)
    save_ckpt("final", args.total_iters - 1)
    print(f"Training complete. Checkpoints in {ckpt_dir}/")

    if envs is not None:
        envs.close()
    writer.close()
