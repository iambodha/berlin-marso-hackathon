"""
WarehouseSort — end-to-end pipeline script (replaces starter.ipynb).

Usage examples
--------------
# 1. Download / stage the demo data (run once):
python run.py setup

# 2. Quick training run to verify the pipeline (~10k iters):
python run.py train --level easy --iters 10000

# 3. Full training run:
python run.py train --level easy --iters 30000 --exp-name warehouse_state_dp_easy

# 4. Evaluate the latest checkpoint:
python run.py eval --level easy

# 5. Evaluate a specific checkpoint:
python run.py eval --level easy --checkpoint il/baselines/diffusion_policy/runs/my_run/checkpoints/best_eval_sort_accuracy.pt

# 6. Train medium / hard levels:
python run.py train --level medium --iters 50000
python run.py train --level hard   --iters 60000
"""

import argparse
import glob
import os
import subprocess
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.abspath(__file__))
DEMO_BASE = os.path.join(ROOT, "il", "demos")
COMPETITION = "marso-hack-berlin-2026-robot-parcel-sorting-challenge"


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _find_checkpoint(level: str, exp_name: str | None) -> str | None:
    pattern = os.path.join(
        ROOT, "il", "baselines", "diffusion_policy", "runs",
        exp_name or f"warehouse_state_dp_{level}",
        "checkpoints", "*.pt",
    )
    ckpts = sorted(glob.glob(pattern))
    if not ckpts:
        # Broader search: any run for this level
        pattern2 = os.path.join(
            ROOT, "il", "baselines", "diffusion_policy", "runs",
            f"*{level}*", "checkpoints", "best_eval_sort_accuracy.pt",
        )
        ckpts = sorted(glob.glob(pattern2), key=os.path.getmtime)
    return ckpts[-1] if ckpts else None


def _demo_h5(level: str) -> str | None:
    pattern = os.path.join(DEMO_BASE, level,
                           "trajectory.*.state.pd_ee_delta_pos.physx_cuda.h5")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def _run(cmd: list[str], cwd: str | None = None) -> int:
    print("\n$ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

def cmd_setup(args):
    """Download and stage demo data into il/demos/<level>/."""
    if args.kaggle_username:
        os.environ["KAGGLE_USERNAME"] = args.kaggle_username
    if args.kaggle_key:
        os.environ["KAGGLE_KEY"] = args.kaggle_key

    from il.download_demos import fetch_and_stage

    try:
        fetch_and_stage(COMPETITION, dest=DEMO_BASE, force=args.force)
    except Exception as e:
        print(f"[setup] download failed: {e}")
        print(
            "\nTo download manually:\n"
            "  1. Join the competition at kaggle.com\n"
            "  2. Set KAGGLE_USERNAME and KAGGLE_KEY env vars (from kaggle.json)\n"
            "  3. Run:  python run.py setup --kaggle-username YOUR_USER --kaggle-key YOUR_KEY\n"
            "  OR place the demo .h5 files under il/demos/<level>/ manually."
        )
        return 1
    return 0


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def cmd_train(args):
    """Train the state Diffusion Policy for a given level."""
    cuda = _has_cuda()
    device_label = "CUDA GPU" if cuda else "CPU (no CUDA detected)"
    print(f"[train] Device: {device_label}")

    if not cuda:
        print(
            "[train] WARNING: CUDA not available. Training will fall back to CPU.\n"
            "         This will be much slower. If you have a GPU, check that your\n"
            "         CUDA drivers and torch are installed correctly.\n"
            "         (pip install torch --index-url https://download.pytorch.org/whl/cu121)"
        )

    # Verify demos exist
    demo_h5 = _demo_h5(args.level)
    if demo_h5 is None:
        print(
            f"[train] Demo dataset not found for level '{args.level}'.\n"
            f"         Run:  python run.py setup  first."
        )
        return 1
    print(f"[train] Using demo: {demo_h5}")

    exp_name = args.exp_name or f"warehouse_state_dp_{args.level}"
    sim_backend = "gpu" if cuda else "cpu"

    # Build the il/train.py hydra command
    cmd = [
        sys.executable,
        os.path.join(ROOT, "il", "train.py"),
        f"method=dp",
        f"demo_dir={args.level}",
        f"sim_backend={sim_backend}",
        f"flags.total_iters={args.iters}",
        f"flags.eval_freq={args.eval_freq}",
        f"flags.exp_name={exp_name}",
        f"flags.batch_size={args.batch_size}",
        f"flags.num_eval_envs={args.num_eval_envs}",
        f"flags.num_eval_episodes={args.num_eval_episodes}",
    ]
    if args.pred_horizon:
        cmd.append(f"flags.pred_horizon={args.pred_horizon}")
    if args.no_cuda:
        cmd.append("flags.cuda=false")
    elif not cuda:
        cmd.append("flags.cuda=false")

    rc = _run(cmd, cwd=ROOT)
    if rc == 0:
        ckpt = _find_checkpoint(args.level, exp_name)
        print(f"\n[train] Done. Best checkpoint: {ckpt or '(none saved yet)'}")
    return rc


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

def cmd_eval(args):
    """Evaluate a trained checkpoint."""
    cuda = _has_cuda()
    device_label = "CUDA GPU" if cuda else "CPU"
    print(f"[eval] Device: {device_label}")

    ckpt = args.checkpoint or _find_checkpoint(args.level, args.exp_name)
    if ckpt is None:
        print(
            f"[eval] No checkpoint found for level '{args.level}'.\n"
            f"       Train first:  python run.py train --level {args.level}\n"
            f"       Or pass:      --checkpoint <path>"
        )
        return 1
    print(f"[eval] Checkpoint: {ckpt}")

    eval_config = os.path.join(ROOT, "conf", "eval", "default.yaml")
    cmd = [
        sys.executable,
        os.path.join(ROOT, "eval.py"),
        f"difficulty={args.level}",
        f"policy=warehouse_sort.il_policy:load_dp",
        f"checkpoint={ckpt}",
        f"eval_config={eval_config}",
    ]
    if not cuda:
        cmd.append("device=cpu")

    return _run(cmd, cwd=ROOT)


# ---------------------------------------------------------------------------
# check — quick sanity check of the environment
# ---------------------------------------------------------------------------

def cmd_check(args):
    """Quick environment sanity check (no GPU required)."""
    import importlib

    cuda = _has_cuda()
    print(f"Python       : {sys.version}")
    print(f"CUDA available: {cuda}")
    if cuda:
        import torch
        print(f"CUDA device  : {torch.cuda.get_device_name(0)}")

    packages = ["torch", "gymnasium", "mani_skill", "warehouse_sort",
                "diffusers", "hydra", "tqdm", "tyro"]
    for pkg in packages:
        try:
            m = importlib.import_module(pkg)
            ver = getattr(m, "__version__", "?")
            print(f"  {pkg:<20} {ver}")
        except ImportError:
            print(f"  {pkg:<20} NOT INSTALLED")

    print("\nDemo datasets staged:")
    for lvl in ("easy", "medium", "hard"):
        h5 = _demo_h5(lvl)
        print(f"  {lvl:<8} {'OK  ' + h5 if h5 else 'MISSING  (run: python run.py setup)'}")

    print("\nCheckpoints:")
    for lvl in ("easy", "medium", "hard"):
        ckpt = _find_checkpoint(lvl, None)
        print(f"  {lvl:<8} {ckpt or '(none)'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="WarehouseSort pipeline: setup | train | eval | check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- setup ---
    p_setup = sub.add_parser("setup", help="Download and stage demo data")
    p_setup.add_argument("--kaggle-username", default=None,
                         help="Kaggle username (or set KAGGLE_USERNAME env var)")
    p_setup.add_argument("--kaggle-key", default=None,
                         help="Kaggle API key (or set KAGGLE_KEY env var)")
    p_setup.add_argument("--force", action="store_true",
                         help="Re-download even if kagglehub has a cached copy")

    # --- train ---
    p_train = sub.add_parser("train", help="Train state Diffusion Policy")
    p_train.add_argument("--level", choices=["easy", "medium", "hard"], default="easy",
                         help="Difficulty level to train on (default: easy)")
    p_train.add_argument("--iters", type=int, default=30000,
                         help="Total training iterations (default: 30000; use 10000 for a quick test)")
    p_train.add_argument("--eval-freq", type=int, default=5000,
                         help="How often to evaluate during training (default: 5000)")
    p_train.add_argument("--batch-size", type=int, default=256,
                         help="Training batch size (default: 256)")
    p_train.add_argument("--num-eval-envs", type=int, default=8,
                         help="Parallel envs during eval (default: 8; reduce if low VRAM)")
    p_train.add_argument("--num-eval-episodes", type=int, default=16,
                         help="Episodes per evaluation (default: 16)")
    p_train.add_argument("--pred-horizon", type=int, default=None,
                         help="Action prediction horizon (default: 16; try 32 for medium/hard)")
    p_train.add_argument("--exp-name", default=None,
                         help="Experiment name (default: warehouse_state_dp_<level>)")
    p_train.add_argument("--no-cuda", action="store_true",
                         help="Force CPU even if CUDA is available")

    # --- eval ---
    p_eval = sub.add_parser("eval", help="Evaluate a trained checkpoint")
    p_eval.add_argument("--level", choices=["easy", "medium", "hard"], default="easy",
                        help="Difficulty level to evaluate (default: easy)")
    p_eval.add_argument("--checkpoint", default=None,
                        help="Path to .pt checkpoint (auto-finds latest if omitted)")
    p_eval.add_argument("--exp-name", default=None,
                        help="Experiment name to find the checkpoint for")

    # --- check ---
    sub.add_parser("check", help="Sanity-check the install and staged data")

    args = parser.parse_args()

    dispatch = {
        "setup": cmd_setup,
        "train": cmd_train,
        "eval":  cmd_eval,
        "check": cmd_check,
    }
    sys.exit(dispatch[args.command](args) or 0)


if __name__ == "__main__":
    main()
