#!/usr/bin/env python3
"""End-to-end WarehouseSort image-IL pipeline (replaces starter.ipynb).

Typical local / Thunder Compute usage:

    # First time on a fresh GPU machine:
    bash scripts/thunder_setup.sh

    # Train + evaluate (defaults: easy demos, 30k iters):
    python run_pipeline.py

    # Quick smoke test (~10 min):
    python run_pipeline.py --total-iters 10000 --exp-name warehouse_rgb_dp_starter

    # Eval only:
    python run_pipeline.py --skip-train --checkpoint path/to/best_eval_sort_accuracy.pt
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_EXP = "warehouse_rgb_dp"
RUNS_DIR = REPO_ROOT / "il" / "baselines" / "diffusion_policy" / "runs"


def configure_headless() -> None:
    os.environ.setdefault("DISPLAY", "")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(f"\n>>> {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, cwd=cwd or REPO_ROOT, check=True)


def check_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU not available. ManiSkill training/eval needs a GPU.\n"
            "On Thunder Compute, create a GPU instance and run scripts/thunder_setup.sh first."
        )
    name = torch.cuda.get_device_name(0)
    print(f"CUDA OK — {name} (torch {torch.__version__})")


LEVELS = ("easy", "medium", "hard")


def resolve_levels(demo_dir: str) -> list[str]:
    """'all' -> all three levels; otherwise a single level or comma-separated list."""
    s = str(demo_dir).strip().lower()
    if s == "all":
        return list(LEVELS)
    return [d.strip() for d in s.split(",") if d.strip()]


def check_demos(demo_dir: str) -> Path:
    levels = resolve_levels(demo_dir)
    first = None
    for lvl in levels:
        demo_h5 = (
            REPO_ROOT / "il" / "demos" / lvl / "trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5"
        )
        demo_json = demo_h5.with_suffix(".json")
        if not demo_h5.exists() or not demo_json.exists():
            raise SystemExit(
                f"Missing demo dataset for level '{lvl}':\n"
                f"  expected: {demo_h5}\n"
                f"  and:      {demo_json}\n\n"
                "Download locally with:  python download_kaggle_data.py\n"
                "Then sync to Thunder:   bash scripts/thunder_sync.sh"
            )
        size_mb = demo_h5.stat().st_size / (1024 * 1024)
        print(f"Demos OK — {lvl} ({size_mb:.0f} MB): {demo_h5.name}")
        first = first or demo_h5
    return first


def smoke_test_env() -> None:
    import gymnasium as gym
    import torch
    import warehouse_sort  # noqa: F401 — registers env
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

    print("Smoke-testing WarehouseSort-v1 on GPU...")
    env = gym.make(
        "WarehouseSort-v1",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pos",
        sim_backend="gpu",
        render_mode="rgb_array",
        difficulty="easy",
        num_parcels=2,
        fixed_poses=True,
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    obs, _ = env.reset(seed=42)
    print(f"  obs keys: {list(obs.keys())}")
    print(f"  rgb:   {tuple(obs['rgb'].shape)} {obs['rgb'].dtype}")
    print(f"  state: {tuple(obs['state'].shape)} {obs['state'].dtype}")
    print(f"  action space: {env.action_space}")
    frame = env.render()
    print(f"  render: {tuple(frame.shape)} on {frame.device}")
    env.close()
    del env
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Environment smoke test passed.")


def train(
    demo_dir: str,
    total_iters: int,
    eval_freq: int,
    exp_name: str,
    *,
    train_only: bool = False,
) -> Path:
    cmd = [
        sys.executable,
        "il/train.py",
        "method=dp_rgb",
        f"demo_dir={demo_dir}",
        f"flags.total_iters={total_iters}",
        f"flags.eval_freq={eval_freq}",
        f"flags.exp_name={exp_name}",
    ]
    if train_only:
        cmd += [
            "flags.skip_env_eval=true",
            "flags.capture_video=false",
            "flags.save_freq=1000",
        ]
    run(cmd)
    ckpt_dir = RUNS_DIR / exp_name / "checkpoints"
    if not ckpt_dir.exists():
        raise SystemExit(f"Training finished but no checkpoints dir: {ckpt_dir}")
    return ckpt_dir


def pick_checkpoint(ckpt_dir: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            raise SystemExit(f"Checkpoint not found: {path}")
        return path

    for name in ("best_eval_sort_accuracy.pt", "latest.pt"):
        candidate = ckpt_dir / name
        if candidate.exists():
            return candidate

    others = sorted(ckpt_dir.glob("*.pt"))
    if others:
        return others[-1]

    raise SystemExit(f"No checkpoint found under {ckpt_dir}")


def evaluate(checkpoint: Path, difficulty: str) -> None:
    run(
        [
            sys.executable,
            "eval.py",
            f"difficulty={difficulty}",
            "policy=warehouse_sort.il_policy:load_dp_rgb",
            f"checkpoint={checkpoint}",
            "eval_config=conf/eval/default.yaml",
        ]
    )


def find_latest_eval_video() -> Path | None:
    videos = sorted(
        glob.glob(str(REPO_ROOT / "outputs" / "**" / "videos" / "*.mp4"), recursive=True),
        key=os.path.getmtime,
    )
    return Path(videos[-1]) if videos else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--demo-dir",
        default="easy",
        help="Training data: a level (easy|medium|hard), a comma-list (easy,medium,hard), "
        "or 'all' to train one policy jointly on every level.",
    )
    parser.add_argument(
        "--difficulty",
        default=None,
        help="Eval difficulty (default: each trained level; 'all' evals easy+medium+hard).",
    )
    parser.add_argument("--total-iters", type=int, default=30_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--exp-name", default=DEFAULT_EXP)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint for eval-only runs")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train on demos only (no sim eval — for hosts without Vulkan, e.g. Thunder prototyping)",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument("--smoke-test-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Which levels to evaluate after training: explicit --difficulty, else every trained level.
    if args.difficulty:
        eval_levels = resolve_levels(args.difficulty)
    else:
        eval_levels = resolve_levels(args.demo_dir)

    configure_headless()
    os.chdir(REPO_ROOT)

    check_cuda()
    check_demos(args.demo_dir)

    if not args.skip_smoke_test and not args.train_only:
        smoke_test_env()
    elif args.train_only:
        print("train-only mode — skipping env smoke test (no Vulkan required).")
    if args.smoke_test_only:
        return

    ckpt_dir = RUNS_DIR / args.exp_name / "checkpoints"

    if not args.skip_train:
        print(
            f"\nTraining RGB Diffusion Policy on '{args.demo_dir}' "
            f"for {args.total_iters} iters (exp_name={args.exp_name})...\n"
        )
        ckpt_dir = train(
            args.demo_dir,
            args.total_iters,
            args.eval_freq,
            args.exp_name,
            train_only=args.train_only,
        )

    if not args.skip_eval and not args.train_only:
        checkpoint = pick_checkpoint(ckpt_dir, args.checkpoint)
        for difficulty in eval_levels:
            print(f"\nEvaluating {checkpoint} on difficulty={difficulty}...\n")
            evaluate(checkpoint, difficulty)
        video = find_latest_eval_video()
        if video:
            print(f"\nEval rollout video: {video}")
        print(f"\nCheckpoints: {ckpt_dir}")
        print(f"TensorBoard:  tensorboard --logdir {RUNS_DIR / args.exp_name}")


if __name__ == "__main__":
    main()
