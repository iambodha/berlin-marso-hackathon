"""Imitation-learning policy entrypoints for eval.py / the judge.

Each function satisfies the policy contract:
    policy.act(obs, deterministic=True) -> Tensor (num_envs, action_dim) in [-1, 1]

Wire one in via the config `policy` field:
    pixi run python eval.py difficulty=easy \\
        policy=warehouse_sort.il_policy:load_dp \\
        checkpoint=<path> eval_config=conf/eval/default.yaml

  load_dp      — state Diffusion Policy (main track; one checkpoint PER level)
  load_dp_rgb  — RGB Diffusion Policy (optional image track; template)
"""

import torch


def _add_baseline_path(rel):
    import os, sys
    p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "il", "baselines", rel))
    if p not in sys.path:
        sys.path.insert(0, p)


# --------------------------------------------------------------------------- #
# State Diffusion Policy (MAIN track) — privileged low-dim state obs. Deployed fully
# closed-loop: re-query every step, execute the first predicted action. The state vector
# is parcel-count-specific, so a checkpoint is trained PER difficulty level.
# --------------------------------------------------------------------------- #
class _DPPolicy:
    def __init__(self, net, scheduler, obs_horizon, pred_horizon, act_dim, device,
                 act_horizon=8, num_inference_steps=16, open_loop=True):
        self.net = net.to(device).eval()
        self.scheduler = scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.pred_horizon = pred_horizon
        self.act_dim = act_dim
        self.device = device
        self.open_loop = open_loop
        # Rolling window of the last `obs_horizon` real observations, matching FrameStack.
        self.hist = None
        self._action_buf = None
        self._buf_idx = 0

    def reset(self):
        """Clear obs history and any cached open-loop action chunk (call on env.reset)."""
        self.hist = None
        self._action_buf = None
        self._buf_idx = 0

    @torch.no_grad()
    def _infer(self, obs_cond):
        B = obs_cond.shape[0]
        naction = torch.randn((B, self.pred_horizon, self.act_dim), device=self.device)
        for k in self.scheduler.timesteps:
            noise_pred = self.net(sample=naction, timestep=k, global_cond=obs_cond)
            naction = self.scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample
        start = self.obs_horizon - 1
        if self.open_loop:
            end = start + self.act_horizon
            return naction[:, start:end].clamp(-1.0, 1.0)
        return naction[:, start:start + 1].clamp(-1.0, 1.0)

    @torch.no_grad()
    def get_action(self, obs):
        """Training-style API: obs is (B, obs_horizon, obs_dim) from FrameStack -> (B, act_horizon, act_dim)."""
        seq = (obs["state"] if isinstance(obs, dict) else obs).float().to(self.device)
        if seq.dim() == 2:
            seq = self._stack_obs(seq)
        obs_cond = seq.flatten(start_dim=1)
        return self._infer(obs_cond)

    def _stack_obs(self, cur):
        """Build (B, obs_horizon, obs_dim) from a single-step (B, obs_dim) vector."""
        if self.hist is None or self.hist[-1].shape != cur.shape:
            self.hist = [cur] * self.obs_horizon
        else:
            self.hist = (self.hist + [cur])[-self.obs_horizon:]
        return torch.stack(self.hist, dim=1)

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        cur = (obs["state"] if isinstance(obs, dict) else obs).float().to(self.device)

        # Always advance the observation window on every env step (including open-loop
        # buffer steps). Previously we returned buffered actions WITHOUT updating hist,
        # so obs_horizon>2 saw stale frames like [o0,o0,o0,o8] instead of a rolling window.
        if cur.dim() == 3:
            obs_cond = cur.flatten(start_dim=1)
        else:
            obs_cond = self._stack_obs(cur).flatten(start_dim=1)

        if self.open_loop and self._action_buf is not None and self._buf_idx < self._action_buf.shape[1]:
            a = self._action_buf[:, self._buf_idx]
            self._buf_idx += 1
            return a

        actions = self._infer(obs_cond)
        if self.open_loop and actions.shape[1] > 1:
            self._action_buf = actions
            self._buf_idx = 1
            return actions[:, 0]
        return actions[:, 0]


def peek_dp_config(checkpoint, obs_dim):
    """Read training horizons from a state-DP checkpoint (saved config + weight auto-detect)."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config") or {}
    sd = ckpt.get("ema_agent", ckpt.get("agent", {}))
    net_sd = {k.replace("noise_pred_net.", "", 1): v for k, v in sd.items()
              if k.startswith("noise_pred_net.")}
    detected_obs = None
    try:
        dsed = net_sd["diffusion_step_encoder.1.weight"].shape[1]
        cond_dim = net_sd["down_modules.0.0.cond_encoder.1.weight"].shape[1]
        g = cond_dim - dsed
        if g % obs_dim == 0 and g > 0:
            detected_obs = g // obs_dim
    except Exception:
        pass
    obs_h = cfg.get("obs_horizon") or detected_obs or 2
    return {
        "obs_horizon": int(obs_h),
        "act_horizon": int(cfg.get("act_horizon", 8)),
        "pred_horizon": int(cfg.get("pred_horizon", 16)),
        "open_loop": bool(cfg.get("open_loop", True)),
        "num_inference_steps": int(cfg.get("num_inference_steps", 16)),
    }


def load_dp(checkpoint, sample_obs, action_space, device,
            obs_horizon=None, act_horizon=8, pred_horizon=16, diffusion_step_embed_dim=64,
            unet_dims=(64, 128, 256), n_groups=8, num_diffusion_iters=100,
            num_inference_steps=16, open_loop=True):
    """Load a state Diffusion Policy checkpoint (uses EMA weights).

    obs_horizon: pass explicitly to override auto-detection (must match checkpoint + env).
    act_horizon / open_loop: match training eval (FrameStack + act_horizon open-loop chunks).
    """
    _add_baseline_path("diffusion_policy")
    from diffusion_policy.conditional_unet1d import ConditionalUnet1D
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    state = sample_obs["state"] if isinstance(sample_obs, dict) else sample_obs
    obs_dim = state.shape[-1]
    act_dim = action_space.shape[0]

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    ckpt_cfg = ckpt.get("config") or {}
    if "act_horizon" in ckpt_cfg:
        act_horizon = int(ckpt_cfg["act_horizon"])
    if "pred_horizon" in ckpt_cfg:
        pred_horizon = int(ckpt_cfg["pred_horizon"])
    if "open_loop" in ckpt_cfg:
        open_loop = bool(ckpt_cfg["open_loop"])
    if "num_inference_steps" in ckpt_cfg:
        num_inference_steps = int(ckpt_cfg["num_inference_steps"])

    sd = ckpt.get("ema_agent", ckpt.get("agent"))
    net_sd = {k.replace("noise_pred_net.", "", 1): v for k, v in sd.items()
              if k.startswith("noise_pred_net.")}

    detected_obs_horizon = None
    try:
        dsed = net_sd["diffusion_step_encoder.1.weight"].shape[1]
        cond_dim = net_sd["down_modules.0.0.cond_encoder.1.weight"].shape[1]
        global_cond_dim = cond_dim - dsed
        if global_cond_dim % obs_dim == 0 and global_cond_dim > 0:
            detected_obs_horizon = global_cond_dim // obs_dim
        diffusion_step_embed_dim = dsed
    except Exception:
        global_cond_dim = None
        dsed = diffusion_step_embed_dim

    if obs_horizon is None:
        obs_horizon = detected_obs_horizon if detected_obs_horizon is not None else 2
    else:
        obs_horizon = int(obs_horizon)
        if detected_obs_horizon is not None and obs_horizon != detected_obs_horizon:
            raise ValueError(
                f"obs_horizon={obs_horizon} but checkpoint implies {detected_obs_horizon} "
                f"(env obs_dim={obs_dim}, global_cond={global_cond_dim}). "
                f"Use difficulty matching training demos and the same obs_horizon."
            )

    expected_cond = obs_horizon * obs_dim
    if global_cond_dim is not None and global_cond_dim != expected_cond:
        raise ValueError(
            f"Obs space mismatch: env gives obs_dim={obs_dim}, obs_horizon={obs_horizon} "
            f"(expect global_cond={expected_cond}) but checkpoint has global_cond={global_cond_dim}. "
            f"Train and eval must use the same difficulty (parcel count) and obs_horizon."
        )

    print(
        f"[load_dp] obs_dim={obs_dim} obs_horizon={obs_horizon} act_horizon={act_horizon} "
        f"pred_horizon={pred_horizon} open_loop={open_loop} global_cond={obs_horizon * obs_dim}",
        flush=True,
    )

    net = ConditionalUnet1D(
        input_dim=act_dim, global_cond_dim=obs_horizon * obs_dim,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=list(unet_dims), n_groups=n_groups,
    )
    net.load_state_dict(net_sd)
    scheduler = DDPMScheduler(num_train_timesteps=num_diffusion_iters,
                              beta_schedule="squaredcos_cap_v2", clip_sample=True,
                              prediction_type="epsilon")
    return _DPPolicy(
        net, scheduler, obs_horizon, pred_horizon, act_dim, device,
        act_horizon=act_horizon, num_inference_steps=num_inference_steps, open_loop=open_loop,
    )


# --------------------------------------------------------------------------- #
# RGB Diffusion Policy (OPTIONAL image track) — image + robot proprioception, NO privileged state.
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
    """Load an RGB (image + state) Diffusion Policy checkpoint (vendored train_rgbd; EMA weights).

    The architecture hyperparameters are read from the checkpoint's saved "config" when present,
    so a checkpoint trained with ANY horizons / encoder / unet size loads correctly with no
    matching args required. The function arguments are only fallback defaults for older
    checkpoints that predate the saved config.
    """
    import types
    import numpy as np
    import gymnasium.spaces as spaces
    _add_baseline_path("diffusion_policy")
    from train_rgbd import Agent

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) or {}
    # checkpoint config wins; fall back to the call-time defaults for legacy checkpoints
    obs_horizon = cfg.get("obs_horizon", obs_horizon)
    act_horizon = cfg.get("act_horizon", act_horizon)
    pred_horizon = cfg.get("pred_horizon", pred_horizon)
    diffusion_step_embed_dim = cfg.get("diffusion_step_embed_dim", diffusion_step_embed_dim)
    unet_dims = cfg.get("unet_dims", unet_dims)
    n_groups = cfg.get("n_groups", n_groups)
    visual_encoder = cfg.get("visual_encoder", visual_encoder)
    num_kp = cfg.get("num_kp", num_kp)

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
