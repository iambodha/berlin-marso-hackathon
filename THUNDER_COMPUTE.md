# Running WarehouseSort on Thunder Compute

This guide replaces the Colab `starter.ipynb` workflow with scripts you can run on a
[Thunder Compute](https://www.thundercompute.com) GPU instance. You already have the demo
data locally — you sync it once, then train on a remote GPU.

---

## What you will run

| File | Purpose |
|------|---------|
| `run_pipeline.py` | Main pipeline: smoke test → train → eval (notebook replacement) |
| `scripts/thunder_sync.sh` | Push code + `il/demos/` to Thunder / pull checkpoints back |
| `scripts/thunder_setup.sh` | One-time pip install on the GPU instance |
| `scripts/thunder_run.sh` | One command to train + evaluate on Thunder |

---

## Step 1 — Install Thunder CLI (on your Mac)

```bash
# macOS: download the .pkg from https://www.thundercompute.com/docs/cli/quickstart
# or Linux:
curl -fsSL https://raw.githubusercontent.com/Thunder-Compute/thunder-cli/main/scripts/install.sh | bash

tnr login          # opens browser — add a payment method in the console first
```

---

## Step 2 — Create a GPU instance

```bash
tnr create
```

Recommended settings for this hackathon:

| Setting | Suggestion |
|---------|------------|
| Mode | **`production`** if you want eval + rollout videos on Thunder; **`development`** (cheaper) for training-only |
| GPU | Any CUDA GPU with ≥16 GB VRAM (e.g. A6000, L4, L40, A100) |
| Template | `base` (Ubuntu) |
| Primary disk | ≥50 GB (demos ~450 MB + checkpoints + PyTorch) |
| Ephemeral disk | Optional — fast scratch at `/ephemeral` if offered |

> **Vulkan / eval requires Production mode.** Thunder **Development** (aka Prototyping) mode is
> CUDA-only — training works, but ManiSkill `eval.py` needs Vulkan for camera rendering and will
> fail with `ErrorIncompatibleDriver` / missing `libEGL_nvidia.so.0`. Either switch modes (below)
> or pull checkpoints and eval on Colab.

Wait until the instance is running:

```bash
tnr status --no-wait
```

Connect once so SSH is configured:

```bash
tnr connect 0
exit
```

---

## Step 3 — Sync your repo + data to Thunder

From your Mac, in the repo root (with `il/demos/easy|medium|hard` already downloaded):

```bash
bash scripts/thunder_sync.sh
```

This uses the SSH host `tnr-0` that the CLI creates. For instance `1`:

```bash
bash scripts/thunder_sync.sh 1
```

---

## Step 4 — Set up the instance (first time only)

```bash
tnr connect 0
cd berlin-marso-hackathon
bash scripts/thunder_setup.sh
```

Takes ~5–10 minutes (PyTorch + ManiSkill). When it finishes:

```bash
source .venv/bin/activate
python run_pipeline.py --smoke-test-only   # quick GPU / env check
```

---

## Step 5 — Train + evaluate

**Full training run (~30–60 min on a good GPU):**

```bash
bash scripts/thunder_run.sh
```

**Quick pipeline test (~10 min, same as the starter notebook):**

```bash
TOTAL_ITERS=10000 EXP_NAME=warehouse_rgb_dp_starter bash scripts/thunder_run.sh
```

**Train on medium/hard:**

```bash
DEMO_DIR=medium EXP_NAME=warehouse_rgb_dp_medium bash scripts/thunder_run.sh
DEMO_DIR=hard   EXP_NAME=warehouse_rgb_dp_hard   bash scripts/thunder_run.sh
```

**Train ONE generalist policy on all levels (recommended for the final score):**

`DEMO_DIR=all` trains jointly on easy+medium+hard. Data augmentation (colour-jitter +
gaussian-blur + 50% horizontal mirror) is on by default (`il/conf/method/dp_rgb.yaml`), and
checkpoints are saved every 1000 iters. After training it auto-evaluates on easy, medium and
hard so you get the full weighted picture.

```bash
DEMO_DIR=all TOTAL_ITERS=30000 EVAL_FREQ=5000 EXP_NAME=warehouse_rgb_dp_all \
  bash scripts/thunder_run.sh
```

If your instance has no Vulkan (Prototyping/Development mode), train without sim eval and pull
checkpoints to eval elsewhere:

```bash
DEMO_DIR=all TOTAL_ITERS=30000 EXP_NAME=warehouse_rgb_dp_all TRAIN_ONLY=1 \
  bash scripts/thunder_run.sh
```

**Direct Python (more control):**

```bash
source .venv/bin/activate
python run_pipeline.py --demo-dir easy --total-iters 30000 --exp-name warehouse_rgb_dp
python run_pipeline.py --skip-train --checkpoint il/baselines/diffusion_policy/runs/warehouse_rgb_dp/checkpoints/best_eval_sort_accuracy.pt
```

---

## Step 6 — Monitor training

On the Thunder instance:

```bash
tensorboard --logdir il/baselines/diffusion_policy/runs --bind_all
```

To view TensorBoard in your browser, expose the port with a
[Cloudflare Tunnel](https://www.thundercompute.com/docs/vscode/operations/port-forwarding)
(or VS Code port forwarding if you use Remote SSH).

Checkpoints land at:

```
il/baselines/diffusion_policy/runs/<exp_name>/checkpoints/
  best_eval_sort_accuracy.pt   ← use this for eval / submission
  latest.pt
```

Eval videos land under `outputs/<date>/<time>/videos/`.

---

## Step 7 — Pull checkpoints back to your Mac

```bash
bash scripts/thunder_sync.sh pull
```

Then evaluate locally (if you have a GPU) or commit checkpoints to your submission repo.

---

## Step 8 — Stop the instance when done

Thunder bills while the instance runs. When you are finished:

- **Development mode:** stop/delete the instance from the console or CLI so you are not charged.
- Consider creating a **snapshot** if you set up a heavy environment you want to reuse.

---

## Eval on Thunder (Vulkan) — Production mode required

If your instance banner says **Mode: Prototyping** (Development mode), Vulkan is **not available**
by design. Switch to Production on your **Mac**:

```bash
tnr modify 0 --mode production --num-gpus 1 -y
```

Production mode only supports **a6000, a100, h100** (not L40). If modify fails or changes your GPU type, your disk persists but you may need to re-run `bash scripts/thunder_setup.sh` after reconnecting.

Wait until the instance is **RUNNING** again (`tnr status`), reconnect (`tnr connect 0`), then:

```bash
cd ~/berlin-marso-hackathon
bash scripts/thunder_fix_vulkan.sh
source .venv/bin/activate
source scripts/thunder_env.sh
python eval.py difficulty=easy \
    policy=warehouse_sort.il_policy:load_dp_rgb \
    checkpoint=il/baselines/diffusion_policy/runs/warehouse_rgb_dp/checkpoints/10000.pt \
    eval_config=conf/eval/default.yaml
```

**Cheaper workflow:** train in Development mode → `bash scripts/thunder_sync.sh pull` on Mac →
eval on [Colab starter.ipynb](starter.ipynb) (T4 GPU).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA GPU not available` | Confirm `nvidia-smi` works; re-run `scripts/thunder_setup.sh` |
| `Missing demo dataset` | Run `bash scripts/thunder_sync.sh` from your Mac again |
| `tnr-0: Host not found` | Run `tnr connect 0` once on Mac to configure SSH (then `exit` — don't run `pull` inside SSH) |
| `bash scripts/thunder_sync.sh pull` fails | Run on **Mac** (`bodha@...` prompt), not inside `ubuntu@thunder-client` |
| Vulkan / `libEGL_nvidia` / `ErrorIncompatibleDriver` | Instance is likely **Prototyping** — run `tnr modify 0 --mode production --num-gpus 1 -y` on Mac, then `bash scripts/thunder_fix_vulkan.sh` |
| Out of disk | Use `--primary-disk 100` when creating the instance; delete old runs |

---

## VS Code / Cursor alternative

1. Install the Thunder Compute extension + Remote SSH.
2. Connect to your instance from the sidebar.
3. Open the synced `berlin-marso-hackathon` folder.
4. Run `scripts/thunder_setup.sh` then `scripts/thunder_run.sh` in the integrated terminal.

---

## Cost tip

- Use **development** mode for experimentation.
- Sync only what you need (`il/demos/easy` alone is ~68 MB if you skip medium/hard for now).
- Pull checkpoints, then **stop the instance** between long idle periods.
