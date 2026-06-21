"""One-stop: evaluate EVERY checkpoint (.pt) in a folder and print sort accuracy. No video.

Builds the simulator ONCE and reuses it across all checkpoints, so scanning a whole folder is
far faster than launching eval.py per file. Architecture (obs_horizon etc.) is auto-detected
from each checkpoint, so mixed settings work.

Usage (run from the repo root):
  python eval_folder.py <folder>
  python eval_folder.py <folder> <difficulty> <n_episodes>
  python eval_folder.py il/baselines/diffusion_policy/runs/warehouse_state_dp_starter/checkpoints
  python eval_folder.py il/.../checkpoints medium 20
  python eval_folder.py il/.../checkpoints hard 20 rgb     # rgb (image) policy instead of state

Args (positional, all optional except folder):
  folder       directory containing .pt checkpoints
  difficulty   easy | medium | hard          (default: medium)
  n_episodes   episodes per checkpoint        (default: 16)
  obs_mode     state | rgb                     (default: state)
"""

import glob
import os
import sys

import torch

from warehouse_sort.utils import compose_cfg, load_agent, make_env, rollout_metrics


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    folder = sys.argv[1]
    difficulty = sys.argv[2] if len(sys.argv) > 2 else "medium"
    n_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    obs_mode = sys.argv[4] if len(sys.argv) > 4 else "state"
    policy = ("warehouse_sort.il_policy:load_dp" if obs_mode == "state"
              else "warehouse_sort.il_policy:load_dp_rgb")

    ckpts = sorted(glob.glob(os.path.join(folder, "*.pt")))
    if not ckpts:
        print(f"no .pt files found in {folder}")
        sys.exit(1)

    cfg = compose_cfg(overrides=[f"difficulty={difficulty}", f"obs_mode={obs_mode}"])
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    seeds = list(range(5000, 5000 + 64))
    pk = dict(cfg.get("policy_kwargs") or {})
    obs_horizon = pk.get("obs_horizon")
    if obs_horizon is None and obs_mode == "state" and ckpts:
        from warehouse_sort.il_policy import peek_dp_config
        P = int(cfg.difficulty.num_parcels)
        obs_dim = 26 + P * 7 + P * 2 + 6 + 4
        peeked = peek_dp_config(ckpts[0], obs_dim)
        obs_horizon = peeked["obs_horizon"]
        pk = {**peeked, **{k: v for k, v in pk.items() if v is not None}}
    obs_horizon = int(obs_horizon or 1)
    from omegaconf import OmegaConf
    pk = OmegaConf.create(pk)

    n_envs = min(max(int(cfg.num_envs), 1), n_episodes)
    print(f"[eval_folder] difficulty={difficulty}  obs_mode={obs_mode}  "
          f"n_episodes={n_episodes}  envs={n_envs}  obs_horizon={obs_horizon}  "
          f"act_horizon={pk.get('act_horizon', '?')}  checkpoints={len(ckpts)}", flush=True)
    env, _ = make_env(cfg, obs_mode, cfg.randomization, num_envs=n_envs, obs_horizon=obs_horizon)

    results = []
    for c in ckpts:
        name = os.path.basename(c)
        try:
            agent, _ = load_agent(c, env, device, entrypoint=policy, policy_kwargs=pk)
            m = rollout_metrics(env, agent, device, n_episodes, seeds, cfg.max_episode_steps)
            acc = m["sort_accuracy"] * 100
            print(f"  {name:32s}  sort_accuracy = {acc:5.1f}%", flush=True)
            results.append((name, acc))
        except Exception as e:
            print(f"  {name:32s}  FAILED: {e}", flush=True)
    env.close()

    if results:
        results.sort(key=lambda r: r[1], reverse=True)
        best_name, best_acc = results[0]
        print("\n=== ranked (best first) ===")
        for name, acc in results:
            print(f"  {acc:5.1f}%   {name}")
        print(f"\nBEST: {best_name}  =  {best_acc:.1f}%")


if __name__ == "__main__":
    main()
