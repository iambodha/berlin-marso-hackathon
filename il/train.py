"""Hydra dispatcher for the IL baselines (consistent with the repo's Hydra style).

This does NOT reimplement any learning logic. It loads a method config (il/conf/method/*.yaml),
converts its `flags` into the vendored baseline script's CLI (underscore keys -> --hyphen-flags;
booleans -> --flag / --no-flag), and runs that real script with the right demo path + shared args.

  pixi run python il/train.py                              # default: dp_rgb (image DP)
  pixi run python il/train.py method=dp_rgb demo_dir=medium
  pixi run python il/train.py method=dp_rgb flags.total_iters=8000 flags.eval_freq=4000

Outputs (checkpoints + tensorboard + eval videos) land under
  il/baselines/<baseline_dir>/runs/<flags.exp_name>/
Evaluate via: pixi run python eval.py ... policy=warehouse_sort.il_policy:load_dp_rgb ...
"""

import os
import subprocess
import sys

import hydra
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))


def _demo_path(demo_dir, kind):
    d = os.path.join(HERE, "demos", demo_dir)
    return os.path.join(d, f"trajectory.{kind}.pd_ee_delta_pos.physx_cuda.h5")


def _resolve_demo_dirs(demo_dir):
    """demo_dir may be a single level ("easy"), a comma list ("easy,medium,hard"), or "all".
    Returns an ordered list of levels; the FIRST level is the "primary" one used to configure
    the eval env (so "all" puts hard first -> in-training eval reflects the hardest level)."""
    s = str(demo_dir).strip().lower()
    if s == "all":
        return ["hard", "medium", "easy"]
    return [d.strip() for d in s.split(",") if d.strip()]


def _flags_to_cli(flags: dict):
    """method.flags (underscore keys) -> vendored tyro CLI args.
    bool True -> --flag, bool False -> --no-flag; everything else -> --flag value."""
    cli = []
    for key, val in flags.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            cli.append(flag if val else "--no-" + key.replace("_", "-"))
        else:
            cli += [flag, str(val)]
    return cli


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg):
    extra_demos = []
    if cfg.demo_path:
        demo = cfg.demo_path
    else:
        dirs = _resolve_demo_dirs(cfg.get("demo_dir", "easy"))
        paths = [_demo_path(d, cfg.demo_kind) for d in dirs]
        for d, p in zip(dirs, paths):
            if not os.path.exists(p):
                sys.exit(f"demo dataset not found: {p}\n  run: pixi run python il/gen_demos.py "
                         f"--difficulty {d}")
        demo, extra_demos = paths[0], paths[1:]   # primary configures the eval env
    if not os.path.exists(demo):
        sys.exit(f"demo dataset not found: {demo}")
    common = ["--env-id", cfg.env_id, "--control-mode", cfg.control_mode,
              "--sim-backend", cfg.sim_backend, "--max-episode-steps", str(cfg.max_episode_steps)]
    flags = OmegaConf.to_container(cfg.flags, resolve=True)
    cmd = [sys.executable, cfg.script, "--demo-path", demo] + common + _flags_to_cli(flags)
    if extra_demos:
        cmd += ["--extra-demo-paths", ",".join(extra_demos)]
    cwd = os.path.join(HERE, "baselines", cfg.baseline_dir)
    method = hydra.core.hydra_config.HydraConfig.get().runtime.choices.get("method", cfg.script)
    print(f"[il/train] method={method}\n"
          f"[il/train] cwd={cwd}\n[il/train] {' '.join(cmd)}", flush=True)
    sys.exit(subprocess.run(cmd, cwd=cwd).returncode)


if __name__ == "__main__":
    main()
