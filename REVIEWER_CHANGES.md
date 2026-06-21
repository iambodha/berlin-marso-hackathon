# Reviewer Guide — Changes vs. Upstream Baseline

**Upstream:** [marso-robotics/berlin-marso-hackathon](https://github.com/marso-robotics/berlin-marso-hackathon)  
**This fork:** [iambodha/berlin-marso-hackathon](https://github.com/iambodha/berlin-marso-hackathon)

This document is for judges and anyone cloning this repo. It lists **every intentional change** from the original hackathon starter, why we made it, and how to run our submission.

Our approach: **state Diffusion Policy** (`warehouse_sort.il_policy:load_dp`) with hyperparameter tuning. We did **not** submit a scripted controller — only a learned diffusion model mapping observations to actions.

---

## Quick start (reproduce our scores)

```bash
pixi install && pixi run install
pixi run python il/download_demos.py   # or use bundled il/demos/

# Easy (2 parcels)
pixi run python eval.py difficulty=easy obs_mode=state \
  policy=warehouse_sort.il_policy:load_dp \
  checkpoint=il/baselines/diffusion_policy/runs/warehouse_easy_obs10_30k/checkpoints/best_eval_sort_accuracy.pt \
  eval_config=conf/eval/default.yaml record_video=false

# Medium (4 parcels)
pixi run python eval.py difficulty=medium obs_mode=state \
  policy=warehouse_sort.il_policy:load_dp \
  checkpoint=il/baselines/diffusion_policy/runs/warehouse_combined_obs8_30k/checkpoints/best_eval_sort_accuracy.pt \
  eval_config=conf/eval/default.yaml record_video=false
```

`submission.yaml` at the repo root points to these checkpoints. **Hard is omitted** — we did not train a 6-parcel checkpoint (state observation size is parcel-count-specific; a medium checkpoint cannot run on hard).

**Note for Windows:** pass `record_video=false` — Vulkan rendering can crash after metrics are printed.

Horizons (`obs_horizon`, `act_horizon`, `open_loop`) are **auto-detected from each checkpoint** — you do not need extra CLI flags.

---

## Submission manifest (`submission.yaml`)

| Level  | Checkpoint run              | Training data        | In-training best sort_accuracy |
|--------|-----------------------------|----------------------|--------------------------------|
| easy   | `warehouse_easy_obs10_30k`  | easy state demos     | 53.1% (iter 20k)               |
| medium | `warehouse_combined_obs8_30k` | medium_combined demos | 20.3% (iter 20k)               |
| hard   | **not submitted**             | —                    | scores 0 (weight 0.5 still applies) |

Policy entrypoint: `warehouse_sort.il_policy:load_dp` (unchanged contract from upstream).

---

## 1. Critical fix: eval pipeline matched training eval

**Problem in upstream:** Training (`il/baselines/diffusion_policy/train.py`) evaluates with `FrameStack(obs_horizon)` and **open-loop execution** of `act_horizon` actions per diffusion query. Upstream `eval.py` stepped one action per `agent.act()` call with a **single-frame** observation and manual 2-frame stacking inside the policy. Scores from `eval.py` did not reflect training-time performance.

**What we changed:**

### `warehouse_sort/utils.py`

- `make_env()` — new `obs_horizon` argument; applies ManiSkill `FrameStack` when `obs_horizon > 1`.
- `rollout_metrics()` — calls `agent.reset()` on each episode reset; if policy has `open_loop` and `get_action()`, executes full action chunks like training eval.
- `load_agent()` — accepts `policy_kwargs` dict and passes it to the policy loader.

### `warehouse_sort/il_policy.py`

- Rebuilt `_DPPolicy` for open-loop action buffering, rolling observation history, and `reset()`.
- Fixed bug: during open-loop buffer steps, observation history was **not** updated (stale frames when `obs_horizon > 2`).
- Added `peek_dp_config()` — reads horizons from checkpoint weights + saved `config` dict.
- `load_dp()` — auto-detects `obs_horizon` from network `global_cond_dim`; reads `act_horizon`, `open_loop`, `num_inference_steps` from checkpoint; validates obs dim vs difficulty.
- `load_dp_rgb()` — reads architecture hyperparameters from checkpoint `config` when present.

### `eval.py`

- Peeks checkpoint for `obs_horizon` when not set in config.
- Passes `policy_kwargs` to loader; builds env with correct `obs_horizon`.
- Added `record_video` flag (skip Vulkan video on fragile setups).

### `conf/config.yaml`

- Added `record_video` and `policy_kwargs` block (`obs_horizon: null` = auto-detect).

---

## 2. Training pipeline (`il/baselines/diffusion_policy/train.py`)

| Addition | Purpose |
|----------|---------|
| `state_aug_noise` | Gaussian noise on state obs during training only (generalization) |
| `eval_inference_steps` | 16-step DDPM at eval time (faster; training still uses 100 steps) |
| Early stopping | On `sort_accuracy`; patience + “not within 5% of best” guard |
| `--resume` | Continue from checkpoint (optimizer, EMA, iteration) |
| Lazy eval envs | Create/destroy eval sim each eval — saves VRAM |
| Checkpoint every eval | Numbered `.pt` at each eval for `eval_folder.py` |
| `config.json` per run | Full hyperparams + `policy_eval` block for eval |
| `results.json` per run | Eval history for sweeps |
| Richer `.pt` format | Saves `config`, `iteration`, optimizer/EMA state |
| Probe env | Short-lived env to read obs space, then freed before training |

### `il/conf/method/dp.yaml`

- Exposes new flags: `eval_inference_steps`, `state_aug_noise`, `early_stop_*`.

### `il/train.py`

- Unbuffered subprocess (`-u`, `PYTHONUNBUFFERED=1`) for live logs on Windows.

---

## 3. New scripts (not in upstream)

| File | Purpose |
|------|---------|
| `eval_folder.py` | Batch-eval all checkpoints in a folder; one shared sim env |
| `sweep.py` | Hyperparameter sweep (lr, obs/act/pred horizons) |
| `run_pipeline.py` | Smoke test → train → eval pipeline |
| `download_kaggle_data.py` | Kaggle API demo download (use env vars for credentials) |
| `render_rollout.py` | Record rollout videos from a checkpoint |
| `il/add_state_to_rgb_demos.py` | Build rgb+privileged-state datasets for image track |
| `THUNDER_COMPUTE.md` + `scripts/thunder_*.sh` | Remote GPU (Thunder Compute) workflow |

---

## 4. RGB / image track (optional, not submitted)

| File | Change |
|------|--------|
| `diffusion_policy/augment.py` | Label-preserving RGB augmentation for `train_rgbd.py` |
| `diffusion_policy/lerobot_encoder.py` | Alternative visual encoder |
| `il/conf/method/dp_rgb.yaml` | RGB+state track config with augmentation |
| `diffusion_policy/utils.py` | Helpers to init Agent without live sim |

We did **not** include an `rgb` block in `submission.yaml` — state track only.

---

## 5. Platform compatibility

| File | Change |
|------|--------|
| `diffusion_policy/make_env.py` | `spawn` multiprocessing on Windows; `reconfiguration_freq=0` on Windows (PhysX GPU crash fix) |
| `eval.py` | `record_video=false` skips Vulkan after metrics |

---

## 6. Environment (`warehouse_sort/env.py`)

- Added `arm_camera` (wrist-mounted Panda camera). **Does not affect state-track eval** — scene camera unchanged.

---

## 7. Data we added or merged

- `il/demos/medium_combined/` — merged medium trajectories (used for best medium checkpoint)
- `il/demos/medium_extra/` — extra medium demos
- `trajectory.rgbstate.*` — via `il/add_state_to_rgb_demos.py` (RGB track experiments)

---

## 8. Our hyperparameters (vs upstream defaults)

Upstream defaults in `il/conf/method/dp.yaml`: `obs_horizon=2`, `act_horizon=8`, `lr=1e-4`, `batch_size=256`.

**Our best runs:**

| Setting | Easy run | Medium run |
|---------|----------|------------|
| `obs_horizon` | 10 | 10 |
| `act_horizon` | 4 | 4 |
| `pred_horizon` | 16 | 16 |
| `lr` | 5e-5 | 5e-5 |
| `batch_size` | 64 | 64 |
| `state_aug_noise` | 0.01 | 0.01 |
| `max_episode_steps` | 400 | 400 |
| `total_iters` | 30k | 30k |
| `eval_freq` | 2000 | 2000 |
| `eval_inference_steps` | 16 | 16 |

Full per-run configs: `il/baselines/diffusion_policy/runs/<exp_name>/config.json`.

---

## 9. What we did **not** change

- `WarehouseSort-v1` reward, sort-accuracy metric, action space
- Diffusion Policy UNet architecture (ConditionalUnet1D)
- Policy contract: `load_fn(checkpoint, obs, action_space, device) → policy.act(obs)`
- Judge interface: `eval.py` + `eval_config` yaml (same as upstream; we fixed behaviour inside it)

---

## 10. Files judges should read first

1. `submission.yaml` — entrypoint + checkpoint paths  
2. `warehouse_sort/il_policy.py` — `load_dp`, `peek_dp_config`  
3. `warehouse_sort/utils.py` — `rollout_metrics`, `make_env`  
4. `eval.py` — eval entry point  
5. `il/baselines/diffusion_policy/runs/*/config.json` — training hyperparameters  

---

## 11. Checkpoint inventory

Committed best checkpoints referenced in `submission.yaml`:

```
il/baselines/diffusion_policy/runs/warehouse_easy_obs10_30k/checkpoints/best_eval_sort_accuracy.pt
il/baselines/diffusion_policy/runs/warehouse_combined_obs8_30k/checkpoints/best_eval_sort_accuracy.pt
```

Additional experimental runs (not in manifest) live under `il/baselines/diffusion_policy/runs/` — see `results.json` in each run folder.
