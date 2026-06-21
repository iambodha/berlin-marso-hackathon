ALGO_NAME = 'BC_Diffusion_state_UNet'

import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from diffusion_policy.evaluate import evaluate

from collections import defaultdict

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader
from diffusion_policy.utils import IterationBasedBatchSampler, worker_init_fn
from diffusion_policy.make_env import make_eval_envs
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from dataclasses import dataclass, field
from typing import Optional, List
import tyro

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

    env_id: str = "PegInsertionSide-v0"
    """the id of the environment"""
    demo_path: str = (
        "demos/PegInsertionSide-v1/trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
    )
    """the path of demo dataset, it is expected to be a ManiSkill dataset h5py format file"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 1024
    """the batch size of sample from the replay memory"""

    # Diffusion Policy specific arguments
    lr: float = 1e-4
    """the learning rate of the diffusion policy"""
    obs_horizon: int = 2 # Seems not very important in ManiSkill, 1, 2, 4 work well
    act_horizon: int = 8 # Seems not very important in ManiSkill, 4, 8, 15 work well
    pred_horizon: int = 16 # 16->8 leads to worse performance, maybe it is like generate a half image; 16->32, improvement is very marginal
    diffusion_step_embed_dim: int = 64 # not very important
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256]) # default setting is about ~4.5M params
    n_groups: int = 8 # jigu says it is better to let each group have at least 8 channels; it seems 4 and 8 are simila
    eval_inference_steps: int = 16
    """DDPM denoising steps used during EVAL rollouts only (training uses the full 100-step
    schedule). Fewer steps = much faster eval feedback with a near-identical signal; matches the
    deployment loader (load_dp uses 16). Set to 100 for full-quality eval, 8-10 for fastest."""
    state_aug_noise: float = 0.0
    """TRAIN-time data augmentation: std of Gaussian noise added to the state observation (in obs
    units) each step. Multiplies the effective data and improves generalisation to the held-out
    wider position randomization. 0 disables; try 0.01-0.03. Eval is never noised."""

    # Environment/experiment specific arguments
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    eval_freq: int = 5000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = None
    """optional extra checkpoint frequency (in addition to one checkpoint saved after every eval).
    None disables these extra saves; eval checkpoints are always kept."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "physx_cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = 'pd_joint_delta_pos'
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""

    resume: Optional[str] = None
    """path to a checkpoint (.pt) to resume training from; training continues from the saved iteration"""

    early_stop_patience: int = 3
    """stop training if the primary eval metric drops for this many consecutive evaluations"""
    early_stop_metric: str = "sort_accuracy"
    """which eval metric to monitor for early stopping (sort_accuracy / success_once / success_at_end)"""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None


class SmallDemoDataset_DiffusionPolicy(Dataset): # Load everything into GPU memory
    def __init__(self, data_path, device, num_traj):
        if data_path[-4:] == '.pkl':
            raise NotImplementedError()
        else:
            from diffusion_policy.utils import load_demo_dataset
            trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
            # trajectories['observations'] is a list of np.ndarray (L+1, obs_dim)
            # trajectories['actions'] is a list of np.ndarray (L, act_dim)

        for k, v in trajectories.items():
            for i in range(len(v)):
                trajectories[k][i] = torch.Tensor(v[i]).to(device)

        # Pre-compute all possible (traj_idx, start, end) tuples, this is very specific to Diffusion Policy
        if 'delta_pos' in args.control_mode or args.control_mode == 'base_pd_joint_vel_arm_pd_joint_vel':
            self.pad_action_arm = torch.zeros((trajectories['actions'][0].shape[1]-1,), device=device)
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
        # else:
        #     raise NotImplementedError(f'Control Mode {args.control_mode} not supported')
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon = args.obs_horizon, args.pred_horizon
        self.slices = []
        num_traj = len(trajectories['actions'])
        total_transitions = 0
        for traj_idx in range(num_traj):
            L = trajectories['actions'][traj_idx].shape[0]
            assert trajectories['observations'][traj_idx].shape[0] == L + 1
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
                (traj_idx, start, start + pred_horizon) for start in range(-pad_before, L - pred_horizon + pad_after)
            ]  # slice indices follow convention [start, end)

        print(f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}")

        self.trajectories = trajectories

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories['actions'][traj_idx].shape

        obs_seq = self.trajectories['observations'][traj_idx][max(0, start):start+self.obs_horizon]
        # start+self.obs_horizon is at least 1
        act_seq = self.trajectories['actions'][traj_idx][max(0, start):end]
        if start < 0: # pad before the trajectory
            obs_seq = torch.cat([obs_seq[0].repeat(-start, 1), obs_seq], dim=0)
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L: # pad after the trajectory
            gripper_action = act_seq[-1, -1]
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end-L, 1)], dim=0)
            # making the robot (arm and gripper) stay still
        assert obs_seq.shape[0] == self.obs_horizon and act_seq.shape[0] == self.pred_horizon
        return {
            'observations': obs_seq,
            'actions': act_seq,
        }

    def __len__(self):
        return len(self.slices)


class Agent(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        self.state_aug_noise = getattr(args, "state_aug_noise", 0.0)
        assert len(env.single_observation_space.shape) == 2 # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1 # (act_dim, )
        assert (env.single_action_space.high == 1).all() and (env.single_action_space.low == -1).all()
        # denoising results will be clipped to [-1,1], so the action should be in [-1,1] as well
        self.act_dim = env.single_action_space.shape[0]

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim, # act_horizon is not used (U-Net doesn't care)
            global_cond_dim=np.prod(env.single_observation_space.shape), # obs_horizon * obs_dim
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2', # has big impact on performance, try not to change
            clip_sample=True, # clip output to [-1,1] to improve stability
            prediction_type='epsilon' # predict noise (instead of denoised action)
        )
        # Use a SHORT denoising schedule for eval rollouts (get_action). This only affects the
        # inference .step() loop; training (compute_loss/add_noise) still samples over all 100
        # timesteps, so the learned model is unchanged -- eval just runs ~6-10x faster.
        self.eval_inference_steps = getattr(args, "eval_inference_steps", self.num_diffusion_iters)
        self.noise_scheduler.set_timesteps(self.eval_inference_steps)

    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq.shape[0]

        # train-time state augmentation: perturb the observation with small Gaussian noise so the
        # policy doesn't overfit exact training poses (helps the held-out wider randomization).
        if self.state_aug_noise > 0:
            obs_seq = obs_seq + torch.randn_like(obs_seq) * self.state_aug_noise

        # observation as FiLM conditioning
        obs_cond = obs_seq.flatten(start_dim=1) # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(
            action_seq, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(
            noisy_action_seq, timesteps, global_cond=obs_cond)

        return F.mse_loss(noise_pred, noise)

    def get_action(self, obs_seq):
        # init scheduler
        # self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        # set_timesteps will change noise_scheduler.timesteps is only used in noise_scheduler.step()
        # noise_scheduler.step() is only called during inference
        # if we use DDPM, and inference_diffusion_steps == train_diffusion_steps, then we can skip this

        # obs_seq: (B, obs_horizon, obs_dim)
        B = obs_seq.shape[0]
        with torch.no_grad():
            obs_cond = obs_seq.flatten(start_dim=1) # (B, obs_horizon * obs_dim)

            # initialize action from Guassian noise
            noisy_action_seq = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_seq.device)

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
        return noisy_action_seq[:, start:end] # (B, act_horizon, act_dim)

def write_run_config(run_name, args, demo_path, env_kwargs, resume_from=None):
    """Persist hyperparameters to runs/<name>/config.json for later reference / eval."""
    import json
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    cfg = {
        "experiment_name": run_name,
        "env_id": args.env_id,
        "control_mode": args.control_mode,
        "sim_backend": args.sim_backend,
        "training": {
            "total_iters": args.total_iters,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_episode_steps": args.max_episode_steps,
            "seed": args.seed,
            "state_aug_noise": getattr(args, "state_aug_noise", 0.0),
        },
        "horizon_params": {
            "obs_horizon": args.obs_horizon,
            "act_horizon": args.act_horizon,
            "pred_horizon": args.pred_horizon,
        },
        "evaluation": {
            "num_eval_envs": args.num_eval_envs,
            "num_eval_episodes": args.num_eval_episodes,
            "eval_freq": args.eval_freq,
            "eval_inference_steps": getattr(args, "eval_inference_steps", 16),
            "early_stop_patience": args.early_stop_patience,
            "early_stop_metric": args.early_stop_metric,
        },
        "logging": {
            "log_freq": args.log_freq,
            "save_freq": args.save_freq,
            "capture_video": args.capture_video,
        },
        "data": {
            "demo_path": os.path.abspath(demo_path),
            "num_demos": args.num_demos,
            "num_parcels": env_kwargs.get("num_parcels"),
            "fixed_poses": env_kwargs.get("fixed_poses"),
            "randomization": env_kwargs.get("randomization"),
        },
        # Pass these to eval.py: policy_kwargs.* (must match training horizons)
        "policy_eval": {
            "obs_horizon": args.obs_horizon,
            "act_horizon": args.act_horizon,
            "pred_horizon": args.pred_horizon,
            "open_loop": True,
            "num_inference_steps": getattr(args, "eval_inference_steps", 16),
        },
    }
    if resume_from:
        cfg["resume_from"] = os.path.abspath(resume_from)
    path = f"runs/{run_name}/config.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[train] Wrote {path}", flush=True)
    return path

def save_ckpt(run_name, tag, iteration=None):
    os.makedirs(f'runs/{run_name}/checkpoints', exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    config = {
        "obs_horizon": args.obs_horizon,
        "act_horizon": args.act_horizon,
        "pred_horizon": args.pred_horizon,
        "open_loop": True,
        "num_inference_steps": getattr(args, "eval_inference_steps", 16),
    }
    torch.save({
        'agent': agent.state_dict(),
        'ema_agent': ema_agent.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'ema_state': ema.state_dict(),
        'iteration': iteration,
        'config': config,
    }, f'runs/{run_name}/checkpoints/{tag}.pt')

if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    if args.demo_path.endswith('.h5'):
        import json
        json_file = args.demo_path[:-2] + 'json'
        with open(json_file, 'r') as f:
            demo_info = json.load(f)
            if 'control_mode' in demo_info['env_info']['env_kwargs']:
                control_mode = demo_info['env_info']['env_kwargs']['control_mode']
            elif 'control_mode' in demo_info['episodes'][0]:
                control_mode = demo_info['episodes'][0]['control_mode']
            else:
                raise Exception('Control mode not found in json')
            assert control_mode == args.control_mode, f"Control mode mismatched. Dataset has control mode {control_mode}, but args has control mode {args.control_mode}"
    # match the eval env to the demo distribution (num parcels, bins, pose randomisation, etc.)
    _demo_scene_kwargs = {}
    if args.demo_path.endswith('.h5') and args.env_id.startswith("WarehouseSort"):
        _dk = demo_info['env_info']['env_kwargs']
        for _k in ("num_parcels", "fixed_poses", "randomization"):
            if _k in _dk:
                _demo_scene_kwargs[_k] = _dk[_k]
    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
    assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(control_mode=args.control_mode, reward_mode="sparse", obs_mode="state", render_mode="rgb_array", human_render_camera_configs=dict(shader_pack="default"))
    env_kwargs.update(_demo_scene_kwargs)   # eval env matches the demos (parcels/bins/randomisation)
    assert args.max_episode_steps != None, "max_episode_steps must be specified as imitation learning algorithms task solve speed is dependent on the data you train on"
    env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    _eval_enabled = args.eval_freq is not None and args.eval_freq > 0
    # Eval envs are created lazily (just before each eval run) and destroyed afterwards so
    # they don't occupy VRAM during training. This prevents OOM on GPUs with limited memory.
    envs = None  # populated inside evaluate_and_save_best when needed

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
            tags=["diffusion_policy"]
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # dataloader setup — built after resume load so remaining_iters is correct
    dataset = SmallDemoDataset_DiffusionPolicy(args.demo_path, device, num_traj=args.num_demos)
    if args.num_demos is None:
        args.num_demos = len(dataset)
    write_run_config(run_name, args, args.demo_path, env_kwargs,
                     resume_from=args.resume if args.resume else None)

    # agent setup
    # When eval is enabled: spin up 1 env briefly to read the real observation/action space
    # (the live env may have a different obs_dim than the HDF5 dataset), then close it
    # immediately so it doesn't occupy VRAM during training.
    # When eval is disabled: derive shapes from the dataset.
    if _eval_enabled:
        print("[train] Creating a temporary env to read observation space...")
        _probe_env = make_eval_envs(
            args.env_id, 1, args.sim_backend, env_kwargs, other_kwargs, video_dir=None
        )
        agent_env_arg = _probe_env
    else:
        import gymnasium as gym
        obs_dim = dataset.trajectories['observations'][0].shape[-1]
        act_dim = dataset.trajectories['actions'][0].shape[-1]
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(args.obs_horizon, obs_dim))
        act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,))
        class _MockEnv:
            single_observation_space = obs_space
            single_action_space = act_space
        agent_env_arg = _MockEnv()
    agent = Agent(agent_env_arg, args).to(device)
    optimizer = optim.AdamW(params=agent.parameters(),
        lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(agent_env_arg, args).to(device)

    # Close and free the probe env — it was only needed for the observation space shape
    if _eval_enabled:
        _probe_env.close()
        del _probe_env
        torch.cuda.empty_cache()
        print("[train] Probe env closed. VRAM freed for training.")

    # Resume from checkpoint if requested
    start_iteration = 0
    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        agent.load_state_dict(ckpt['agent'])
        ema_agent.load_state_dict(ckpt['ema_agent'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        if 'ema_state' in ckpt:
            ema.load_state_dict(ckpt['ema_state'])
        start_iteration = (ckpt.get('iteration') or 0) + 1
        print(f"Resuming from iteration {start_iteration} / {args.total_iters}")

    remaining_iters = args.total_iters - start_iteration
    if remaining_iters <= 0:
        print(f"Nothing to train: checkpoint is already at iteration {start_iteration} >= total_iters {args.total_iters}")
        if envs is not None:
            envs.close()
        writer.close()
        raise SystemExit(0)
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, remaining_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
    )

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    # Early-stop state (mutable dict so the closure can write to it)
    _es = {
        'history': [],       # list of (iteration, metric_value)
        'best_value': -1.0,
        'best_iter': 0,
        'should_stop': False,
    }

    # _eval_enabled is set earlier when creating eval envs

    # define evaluation and logging functions
    def evaluate_and_save_best(iteration):
        if not _eval_enabled:
            return
        if iteration % args.eval_freq == 0 and iteration > 0:
            last_tick = time.time()
            ema.copy_to(ema_agent.parameters())

            # Create eval envs just-in-time so they don't occupy VRAM during training
            print(f"\n[eval] Creating eval environments (iter {iteration})...")
            _eval_envs = make_eval_envs(
                args.env_id, args.num_eval_envs, args.sim_backend,
                env_kwargs, other_kwargs, video_dir=None,
            )
            try:
                eval_metrics = evaluate(
                    args.num_eval_episodes, ema_agent, _eval_envs, device, args.sim_backend
                )
            finally:
                _eval_envs.close()
                del _eval_envs
                torch.cuda.empty_cache()

            timings["eval"] += time.time() - last_tick

            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                print(f"{k}: {eval_metrics[k]:.4f}")

            # Keep a numbered checkpoint for every eval so eval_folder.py can scan the full run.
            save_ckpt(run_name, str(iteration), iteration=iteration)
            print(f"Saved checkpoint at iteration {iteration}.")

            save_on_best_metrics = ["sort_accuracy", "success_once", "success_at_end"]
            for k in save_on_best_metrics:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    save_ckpt(run_name, f"best_eval_{k}", iteration=iteration)
                    print(
                        f"New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint."
                    )

            # --- Early stopping ---
            primary = eval_metrics.get(args.early_stop_metric)
            if primary is not None:
                _es['history'].append((iteration, primary))
                if primary > _es['best_value']:
                    _es['best_value'] = primary
                    _es['best_iter'] = iteration

                # Stop only when BOTH conditions hold:
                #  1. The last `patience` evals are strictly decreasing (no blips)
                #  2. None of those evals are within 5% of the all-time best
                #     (a near-best value means the model is still competitive)
                if len(_es['history']) >= args.early_stop_patience + 1:
                    tail = _es['history'][-args.early_stop_patience:]
                    tail_vals = [v for _, v in tail]
                    threshold = _es['best_value'] * 0.95

                    all_decreasing = all(
                        tail_vals[i] > tail_vals[i + 1]
                        for i in range(len(tail_vals) - 1)
                    )
                    none_near_best = all(v < threshold for v in tail_vals)

                    if all_decreasing and none_near_best:
                        print(
                            f"\n[early stop] '{args.early_stop_metric}' has been strictly "
                            f"decreasing for {args.early_stop_patience} consecutive evals "
                            f"and none are within 5% of the best ({_es['best_value']:.4f})."
                        )
                        print(
                            f"[early stop] Best was {_es['best_value']:.4f} "
                            f"at iteration {_es['best_iter']} "
                            f"(checkpoint: runs/{run_name}/checkpoints/best_eval_{args.early_stop_metric}.pt)"
                        )
                        _es['should_stop'] = True
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
    for i, data_batch in enumerate(train_dataloader):
        iteration = start_iteration + i
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

        # Evaluation
        evaluate_and_save_best(iteration)
        log_metrics(iteration)

        # Optional extra checkpoints between evals (eval saves happen in evaluate_and_save_best).
        if args.save_freq is not None and iteration > 0 and iteration % args.save_freq == 0:
            save_ckpt(run_name, str(iteration), iteration=iteration)
        pbar.update(1)
        pbar.set_postfix({"loss": total_loss.item()})
        last_tick = time.time()

        if _es['should_stop']:
            print("\n[early stop] Halting training early.")
            break

    else:
        # Loop completed without early stop — run a final eval
        evaluate_and_save_best(args.total_iters)
        log_metrics(args.total_iters)

    # Print eval summary
    if _es['history']:
        print("\n=== Eval history ===")
        for it, val in _es['history']:
            marker = " <-- best" if it == _es['best_iter'] else ""
            print(f"  iter {it:>6}: {args.early_stop_metric} = {val:.4f}{marker}")
        print(f"Best checkpoint: runs/{run_name}/checkpoints/best_eval_{args.early_stop_metric}.pt")

    # always save a final checkpoint so there is always something to evaluate later
    save_ckpt(run_name, "latest", iteration=args.total_iters - 1)
    print(f"Saved final checkpoint: runs/{run_name}/checkpoints/latest.pt")

    # Write machine-readable results for sweep scripts
    import json as _json
    _results = {
        "exp_name": run_name,
        "eval_history": [[int(it), float(val)] for it, val in _es["history"]],
        "best_value": float(_es["best_value"]),
        "best_iter": int(_es["best_iter"]),
        "early_stopped": bool(_es["should_stop"]),
        "metric": args.early_stop_metric,
    }
    _results_path = f"runs/{run_name}/results.json"
    with open(_results_path, "w") as _f:
        _json.dump(_results, _f, indent=2)
    print(f"Results written to {_results_path}")

    _cfg_path = f"runs/{run_name}/config.json"
    if os.path.isfile(_cfg_path):
        with open(_cfg_path) as _f:
            _run_cfg = _json.load(_f)
        _run_cfg["final_results"] = _results
        with open(_cfg_path, "w") as _f:
            _json.dump(_run_cfg, _f, indent=2)

    writer.close()
