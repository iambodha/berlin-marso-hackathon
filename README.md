# WarehouseSort · Berlin Marso hackathon

Our **Team Waitlist** submission for a simulated robot parcel-sorting challenge: train a state-based Diffusion Policy to move parcels into bins matching their color. This source release includes our policy integration, training/evaluation changes, experiment configurations, and saved evaluation histories.

The main engineering contribution was making evaluation match training: observation stacking, action-chunk execution, checkpoint configuration, and episode resets must agree before scores are comparable.

**[Our changes](REVIEWER_CHANGES.md)** · **[Saved experiments](il/baselines/diffusion_policy/runs/)** · **[Original challenge](CHALLENGE.md)** · **[Upstream project](https://github.com/marso-robotics/berlin-marso-hackathon)**

## What we built on top of the starter

- Matched standalone evaluation to training's stacked observations and open-loop action chunks.
- Fixed stale observation history during buffered actions and reset policy state between episodes.
- Read horizons and architecture settings from checkpoint metadata instead of requiring matching manual flags.
- Added training resumes, state-noise augmentation, early stopping, checkpoint evaluation, and experiment sweeps.
- Added Windows compatibility changes and a schematic state-rollout renderer that avoids Vulkan video capture.

The environment, reward, task definition, and Diffusion Policy UNet originate upstream. This is a team submission; the repository does not assign individual ownership of every change. The detailed [reviewer guide](REVIEWER_CHANGES.md) separates our additions from the baseline.

## Results and their limits

| Difficulty | Recorded result | Evidence |
| --- | --- | --- |
| Easy | **53.1%** best training-time sort accuracy, reported at iteration 20k | Historical reviewer notes; the corresponding evaluation history and weights are not bundled |
| Medium | **20.3125%** best training-time sort accuracy at iteration 20k; **18.75%** at iteration 30k | [Committed evaluation history](il/baselines/diffusion_policy/runs/warehouse_combined_obs8_30k/results.json) |
| Hard | No trained checkpoint submitted, according to the reviewer guide | Omitted from the corrected submission manifest |

These are training-time evaluation results, **not a held-out competition score**. Sort accuracy measures correctly placed parcels, not the proportion of fully solved episodes. No aggregate competition score is asserted.

You can inspect the saved medium result with **Python 3.10+ and no extra packages**:

```sh
python scripts/report_results.py il/baselines/diffusion_policy/runs/warehouse_combined_obs8_30k/results.json
python -m unittest discover -s tests -v
```

## Run the full learning system

The committed Pixi environment targets **Linux x86-64**. Training/evaluation use PyTorch, ManiSkill, SAPIEN, Hydra, and a suitable simulation/GPU environment. The full GPU environment was not revalidated during this documentation pass.

```sh
pixi install
pixi run python il/download_demos.py
pixi run python il/train.py method=dp demo_dir=easy
```

Dataset access requires joining the upstream Kaggle competition and configuring credentials locally. See [CHALLENGE.md](CHALLENGE.md) and [il/README.md](il/README.md) for the supplied data and baseline training flow. Consult the saved `config.json` files and reviewer guide to reproduce our tuned settings; the command above starts the baseline training flow.

**Trained weights are not included or currently available from the maintainer.** The paths in `submission.yaml` identify historical run outputs, not downloadable artifacts. A fresh checkout cannot reproduce those policy scores or render our policy without retraining and saving compatible weights. The old manifest's unsupported hard checkpoint has been removed.

## Produce a model rollout after training

With a compatible checkpoint saved on your machine:

```sh
pixi run python render_rollout.py \
  --checkpoint /path/to/your/best_eval_sort_accuracy.pt \
  --difficulty easy --seed 5000 --fps 20 \
  --output videos/learned-easy.gif
```

The renderer creates a **schematic visualization from an actual policy rollout**, using parcel/bin/TCP state rather than photorealistic simulator footage. It reports that episode's sorted-parcel count. Existing output paths are refused; use a new filename. `--help` and missing-checkpoint errors work without importing the GPU stack.

The videos in `media/` and `scripted/` are the organizers' **scripted demonstration policy**, used to explain the task. They do not show our learned policy and are not evidence for the scores above. No learned-policy clip is claimed in this release.

## Repository guide

| Path | Purpose |
| --- | --- |
| `warehouse_sort/il_policy.py` | Policy loading, checkpoint settings, observation history, action buffering |
| `warehouse_sort/utils.py`, `eval.py` | Environment setup and evaluation execution |
| `il/baselines/diffusion_policy/` | Training code and saved experiment configurations/results |
| `sweep.py`, `eval_folder.py`, `run_pipeline.py` | Experiment and checkpoint evaluation tools |
| `render_rollout.py`, `warehouse_sort/viz.py` | Schematic rollout rendering |
| `submission.yaml` | Historical easy/medium checkpoint locations; external weights required |
| `scripts/report_results.py`, `tests/` | CPU-only result inspection and renderer-control regression checks |

Source checks cover result extraction, termination/step-limit handling, and the renderer CLI. They do not validate simulation physics, train a model, or certify task performance. Preserve upstream attribution and the original challenge/submission materials when reusing the system; this maintenance pass does not assign a new license to upstream code or datasets.
