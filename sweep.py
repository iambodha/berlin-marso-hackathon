"""Hyperparameter sweep for Diffusion Policy on WarehouseSort-v1 (medium).

Runs each config sequentially. After every run:
  - The full run directory is copied to  il/baselines/diffusion_policy/runs/saves/saveN/
  - A row is appended to  sweep_results.csv  (in the repo root)

Resume-safe: the script counts existing saveN directories so it never overwrites
a completed run if you restart after a crash.

Usage:
    python sweep.py                  # run all configs
    python sweep.py --start C4      # skip configs before C4
    python sweep.py --only C3 C7    # run only these config IDs
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config table  (steps column is intentionally ignored — fixed at 30 000)
# ---------------------------------------------------------------------------
CONFIGS = [
    {"id": "C1",  "lr": 0.0001, "obs": 2, "act": 4, "pred": 16},
    {"id": "C2",  "lr": 0.0001, "obs": 2, "act": 2, "pred": 16},
    {"id": "C3",  "lr": 0.0003, "obs": 2, "act": 2, "pred": 16},
    {"id": "C4",  "lr": 0.0001, "obs": 4, "act": 4, "pred": 16},
    {"id": "C5",  "lr": 0.0001, "obs": 4, "act": 2, "pred": 16},
    {"id": "C6",  "lr": 0.0002, "obs": 2, "act": 4, "pred": 16},
    {"id": "C7",  "lr": 0.0002, "obs": 2, "act": 2, "pred": 12},
    {"id": "C8",  "lr": 0.0001, "obs": 2, "act": 4, "pred": 16},
    {"id": "C9",  "lr": 0.0003, "obs": 3, "act": 2, "pred": 16},
    {"id": "C10", "lr": 0.0001, "obs": 2, "act": 2, "pred": 8},
    {"id": "C11", "lr": 0.0001, "obs": 3, "act": 4, "pred": 24},
    {"id": "C12", "lr": 0.0002, "obs": 4, "act": 2, "pred": 16},
]

# ---------------------------------------------------------------------------
# Fixed training settings
# ---------------------------------------------------------------------------
TOTAL_ITERS         = 10_000
EVAL_FREQ           = 1_000
SAVE_FREQ           = 1_000
LOG_FREQ            = 500
BATCH_SIZE          = 128
NUM_EVAL_ENVS       = 8
NUM_EVAL_EPISODES   = 16
DEMO_DIR            = "medium"
EARLY_STOP_PATIENCE = 5   # consecutive strictly-decreasing evals (none within 5% of best)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE      = Path(__file__).parent.resolve()
RUNS_DIR  = HERE / "il" / "baselines" / "diffusion_policy" / "runs"
SAVES_DIR = RUNS_DIR          # saveN directories live directly inside runs/
CSV_PATH  = HERE / "sweep_results.csv"

CSV_FIELDS = [
    "save_name", "config_id", "lr", "obs_horizon", "act_horizon", "pred_horizon",
    "best_iter", "best_sort_accuracy", "early_stopped", "total_evals", "exp_name",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def next_save_index() -> int:
    """Return the next free saveN index by scanning runs/ for existing saveN dirs."""
    SAVES_DIR.mkdir(parents=True, exist_ok=True)
    indices = []
    for d in SAVES_DIR.iterdir():
        if d.is_dir() and d.name.startswith("save"):
            try:
                indices.append(int(d.name[4:]))   # "save" is 4 chars
            except ValueError:
                pass
    idx = max(indices, default=0) + 1
    print(f"[sweep] Existing saves: {sorted(indices)}  →  next index: {idx}")
    return idx


def already_run(config_id: str) -> bool:
    """True if this config already has a row in sweep_results.csv."""
    if not CSV_PATH.exists():
        return False
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("config_id") == config_id:
                return True
    return False


def run_config(cfg: dict, save_idx: int) -> dict:
    exp_name  = f"sweep_{cfg['id']}"
    save_name = f"save{save_idx}"

    cmd = [
        sys.executable, "-u", "il/train.py",
        "method=dp",
        f"demo_dir={DEMO_DIR}",
        f"flags.total_iters={TOTAL_ITERS}",
        f"flags.batch_size={BATCH_SIZE}",
        f"flags.lr={cfg['lr']}",
        f"flags.obs_horizon={cfg['obs']}",
        f"flags.act_horizon={cfg['act']}",
        f"flags.pred_horizon={cfg['pred']}",
        f"flags.eval_freq={EVAL_FREQ}",
        f"flags.save_freq={SAVE_FREQ}",
        f"flags.log_freq={LOG_FREQ}",
        f"flags.num_eval_envs={NUM_EVAL_ENVS}",
        f"flags.num_eval_episodes={NUM_EVAL_EPISODES}",
        f"flags.early_stop_patience={EARLY_STOP_PATIENCE}",
        f"flags.exp_name={exp_name}",
    ]

    print(f"\n{'='*65}")
    print(f"  [{save_name}]  Config {cfg['id']}  →  exp: {exp_name}")
    print(f"  lr={cfg['lr']}  obs={cfg['obs']}  act={cfg['act']}  pred={cfg['pred']}")
    print(f"  total_iters={TOTAL_ITERS}  eval_freq={EVAL_FREQ}  save_freq={SAVE_FREQ}")
    print(f"{'='*65}\n")

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(HERE))
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"\n  [sweep] ERROR: training subprocess exited with code {proc.returncode}.")
        print(f"  [sweep] Skipping save/CSV for {cfg['id']} — fix the error and re-run with --only {cfg['id']} --force")
        return None

    # Parse results.json written by train.py
    results_path = RUNS_DIR / exp_name / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            res = json.load(f)
        best_iter  = res.get("best_iter", -1)
        best_val   = res.get("best_value", 0.0)
        early_stop = res.get("early_stopped", False)
        n_evals    = len(res.get("eval_history", []))
    else:
        print(f"  [sweep] WARNING: results.json not found for {exp_name} — logging zeros")
        best_iter, best_val, early_stop, n_evals = -1, 0.0, False, 0

    # Copy run directory into saves/saveN
    src = RUNS_DIR / exp_name
    dst = SAVES_DIR / save_name
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  [sweep] Copied run → {dst.relative_to(HERE)}")
    else:
        print(f"  [sweep] WARNING: run directory {src} not found — nothing copied")

    row = {
        "save_name":          save_name,
        "config_id":          cfg["id"],
        "lr":                 cfg["lr"],
        "obs_horizon":        cfg["obs"],
        "act_horizon":        cfg["act"],
        "pred_horizon":       cfg["pred"],
        "best_iter":          best_iter,
        "best_sort_accuracy": f"{best_val:.4f}",
        "early_stopped":      early_stop,
        "total_evals":        n_evals,
        "exp_name":           exp_name,
    }

    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(
        f"\n  [sweep] {cfg['id']} finished in {elapsed/60:.1f} min  |  "
        f"best sort_accuracy = {best_val:.4f} @ iter {best_iter}  |  "
        f"early_stopped = {early_stop}"
    )
    return row


def print_leaderboard():
    if not CSV_PATH.exists():
        return
    rows = []
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    rows_sorted = sorted(rows, key=lambda r: float(r["best_sort_accuracy"]), reverse=True)
    print(f"\n{'='*65}")
    print("  LEADERBOARD (all completed configs, sorted by best sort_accuracy)")
    print(f"  {'Rank':<5} {'Save':<8} {'Config':<6} {'lr':<8} {'obs':<4} {'act':<4} {'pred':<5} {'best_acc':<10} {'iter':<7} {'stopped'}")
    print(f"  {'-'*60}")
    for rank, r in enumerate(rows_sorted, 1):
        print(
            f"  {rank:<5} {r['save_name']:<8} {r['config_id']:<6} {r['lr']:<8} "
            f"{r['obs_horizon']:<4} {r['act_horizon']:<4} {r['pred_horizon']:<5} "
            f"{r['best_sort_accuracy']:<10} {r['best_iter']:<7} {r['early_stopped']}"
        )
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",  metavar="ID", help="skip all configs before this ID")
    parser.add_argument("--only",   metavar="ID", nargs="+", help="run only these config IDs")
    parser.add_argument("--force",  action="store_true", help="re-run configs already in CSV")
    args = parser.parse_args()

    configs = CONFIGS
    if args.only:
        configs = [c for c in configs if c["id"] in args.only]
    elif args.start:
        ids = [c["id"] for c in configs]
        if args.start not in ids:
            sys.exit(f"Unknown config id: {args.start}")
        configs = configs[ids.index(args.start):]

    save_idx = next_save_index()
    completed = []

    for cfg in configs:
        if not args.force and already_run(cfg["id"]):
            print(f"\n[sweep] Skipping {cfg['id']} — already in {CSV_PATH.name} (use --force to re-run)")
            continue

        row = run_config(cfg, save_idx)
        if row is None:
            print(f"  [sweep] Stopping sweep due to failed run for {cfg['id']}.")
            break
        completed.append(row)
        save_idx += 1
        print_leaderboard()

    if completed:
        print(f"\n[sweep] Done. {len(completed)} config(s) run. Results in {CSV_PATH.name}")
    else:
        print("\n[sweep] No configs were run.")


if __name__ == "__main__":
    main()
