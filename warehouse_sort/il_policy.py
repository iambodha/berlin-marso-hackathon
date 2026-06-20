"""Imitation-learning policy entrypoints for eval.py / the judge.

Each function satisfies the policy contract:
    policy.act(obs, deterministic=True) -> Tensor (num_envs, action_dim) in [-1, 1]

Wire one in via the config `policy` field:
    pixi run python eval.py difficulty=easy \\
        policy=warehouse_sort.il_policy:load_dp_rgb \\
        checkpoint=<path> eval_config=conf/eval/default.yaml
"""

import torch


def _add_baseline_path(rel):
    import os, sys
    p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "il", "baselines", rel))
    if p not in sys.path:
        sys.path.insert(0, p)


# --------------------------------------------------------------------------- #
# RGB Diffusion Policy — image + robot proprioception, NO privileged state.
# Same fixed image input shape at every difficulty; same checkpoint runs across configs.
# Template only — image IL is not yet solving this task.
# --------------------------------------------------------------------------- #
class _DPRgbPolicy:
    def __init__(self, agent, obs_horizon, device, num_inference_steps=16):
        self.agent = agent.to(device).eval()
        self.agent.noise_scheduler.set_timesteps(num_inference_steps)
        self.obs_horizon = obs_horizon
        self.device = device
        self.prev = None

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        state = obs["state"].float().to(self.device)
        rgb = obs["rgb"].to(self.device)
        cur = {"state": state, "rgb": rgb}
        if self.prev is None or self.prev["state"].shape != state.shape:
            self.prev = cur
        obs_seq = {
            "state": torch.stack([self.prev["state"], state], dim=1),
            "rgb": torch.stack([self.prev["rgb"], rgb], dim=1),
        }
        self.prev = cur
        aseq = self.agent.get_action(obs_seq)
        return aseq[:, 0].clamp(-1.0, 1.0)


def load_dp_rgb(checkpoint, sample_obs, action_space, device,
                obs_horizon=2, act_horizon=8, pred_horizon=16,
                diffusion_step_embed_dim=64, unet_dims=(64, 128, 256), n_groups=8,
                num_inference_steps=16, visual_encoder="resnet18", num_kp=32):
    """Load an RGB Diffusion Policy checkpoint (vendored train_rgbd; uses EMA weights).

    Template implementation — image IL is not yet solving this task.
    """
    import types
    import numpy as np
    import gymnasium.spaces as spaces
    _add_baseline_path("diffusion_policy")
    from train_rgbd import Agent

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    # Auto-match the architecture to whatever was trained: train_rgbd saves vars(args) under
    # "args", so a checkpoint trained with any encoder/horizon loads correctly here with no
    # extra flags from eval.py / the judge.
    ck = ckpt.get("args", {}) or {}
    def _a(key, default):
        v = ck.get(key, default)
        return default if v is None else v
    obs_horizon = int(_a("obs_horizon", obs_horizon))
    act_horizon = int(_a("act_horizon", act_horizon))
    pred_horizon = int(_a("pred_horizon", pred_horizon))
    diffusion_step_embed_dim = int(_a("diffusion_step_embed_dim", diffusion_step_embed_dim))
    unet_dims = list(_a("unet_dims", unet_dims))
    n_groups = int(_a("n_groups", n_groups))
    visual_encoder = _a("visual_encoder", visual_encoder)
    num_kp = int(_a("num_kp", num_kp))

    h, w, c = sample_obs["rgb"].shape[1:]
    state_dim = sample_obs["state"].shape[1]
    stub = types.SimpleNamespace(
        single_observation_space=spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, (obs_horizon, state_dim), np.float32),
            "rgb": spaces.Box(0, 255, (obs_horizon, h, w, c), np.uint8),
        }),
        single_action_space=spaces.Box(-1.0, 1.0, (action_space.shape[0],), np.float32),
    )
    args = types.SimpleNamespace(
        obs_horizon=obs_horizon, act_horizon=act_horizon, pred_horizon=pred_horizon,
        diffusion_step_embed_dim=diffusion_step_embed_dim, unet_dims=list(unet_dims),
        n_groups=n_groups, visual_encoder=visual_encoder, num_kp=num_kp,
    )
    agent = Agent(stub, args)
    agent.load_state_dict(ckpt.get("ema_agent", ckpt.get("agent")))
    return _DPRgbPolicy(agent, obs_horizon, device, num_inference_steps=num_inference_steps)
