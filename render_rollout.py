"""Roll out a state-DP checkpoint and save a schematic 3D video (Windows-safe).

Uses the privileged state vector each step — no Vulkan / rgb_array offscreen rendering.
This is NOT photorealistic SAPIEN footage; it shows parcels, bins, and TCP in 3D.

Example:
  pixi run python render_rollout.py \\
      --checkpoint il/baselines/diffusion_policy/runs/warehouse_state_dp_easy/checkpoints/best_eval_sort_accuracy.pt \\
      --difficulty easy \\
      --output videos/easy_rollout.gif
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from omegaconf import OmegaConf

import warehouse_sort  # noqa: F401
from warehouse_sort.il_policy import peek_dp_config
from warehouse_sort.utils import compose_cfg, load_agent, make_env, to_device
from warehouse_sort.viz import parse_state_obs, save_state_rollout_video


def _rollout_frames(env, agent, device, max_steps, num_parcels, seed):
    obs, _ = env.reset(seed=seed)
    if hasattr(agent, "reset"):
        agent.reset()
    obs = to_device(obs, device)
    frames = [parse_state_obs(obs, num_parcels)]
    steps = 0

    while steps < max_steps - 1:
        if getattr(agent, "open_loop", False) and hasattr(agent, "get_action"):
            action_seq = agent.get_action(obs)
            for i in range(action_seq.shape[1]):
                obs, _, _, truncated, _ = env.step(action_seq[:, i])
                obs = to_device(obs, device)
                steps += 1
                frames.append(parse_state_obs(obs, num_parcels))
                if truncated.any() or steps >= max_steps - 1:
                    return frames
        else:
            obs, _, _, truncated, _ = env.step(agent.act(obs, deterministic=True))
            obs = to_device(obs, device)
            steps += 1
            frames.append(parse_state_obs(obs, num_parcels))
            if truncated.any():
                break
    return frames


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    p.add_argument("--output", default=None, help="Output .gif or .mp4 (default: ./videos/<name>.gif)")
    p.add_argument("--seed", type=int, default=5000)
    p.add_argument("--max-steps", type=int, default=None, help="Rollout length (default: from config)")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--obs-horizon", type=int, default=None, help="Override auto-detect from checkpoint")
    p.add_argument("--policy", default="warehouse_sort.il_policy:load_dp")
    args = p.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    cfg = compose_cfg(overrides=[f"difficulty={args.difficulty}", f"checkpoint={args.checkpoint}"])
    cfg.policy = args.policy
    max_steps = int(args.max_steps or cfg.max_episode_steps)
    randomization = cfg.randomization
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    num_parcels = int(cfg.difficulty.num_parcels)
    obs_dim = 26 + num_parcels * 7 + num_parcels * 2 + 6 + 4
    peeked = peek_dp_config(args.checkpoint, obs_dim)
    obs_horizon = int(args.obs_horizon or peeked["obs_horizon"])
    pk = OmegaConf.create({**peeked, **OmegaConf.to_container(cfg.get("policy_kwargs") or {})})
    pk.obs_horizon = obs_horizon

    print(
        f"[render_rollout] difficulty={args.difficulty} num_parcels={num_parcels} "
        f"obs_horizon={obs_horizon} max_steps={max_steps} device={device}",
        flush=True,
    )

    env, _ = make_env(
        cfg, cfg.obs_mode, randomization,
        num_envs=1, obs_horizon=obs_horizon, render_mode=None,
    )
    agent, _ = load_agent(args.checkpoint, env, device, entrypoint=cfg.policy, policy_kwargs=pk)

    frames = _rollout_frames(env, agent, device, max_steps, num_parcels, args.seed)
    ev = env.unwrapped.evaluate()
    sorted_n = float(ev["success_count"][0].item())
    env.close()

    ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out = args.output or os.path.join("videos", f"{args.difficulty}_{ckpt_stem}.gif")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)

    title = f"{args.difficulty}  sorted={sorted_n:.0f}/{num_parcels}"
    saved = save_state_rollout_video(frames, out, fps=args.fps, title=title)
    print(f"[render_rollout] {len(frames)} frames -> {saved}", flush=True)
    print(f"[render_rollout] sort_accuracy (episode): {sorted_n / num_parcels * 100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
