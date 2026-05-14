# SCAR: Self-Supervised Continuous Action Representation Learning

This repository contains the cleaned SCAR code path for learning continuous
latent action representations from visual transitions and using them as a
transferable conditioning interface for a Wan-based forward dynamics model.

The legacy package and deprecated CLAM code are intentionally not part
of this tree. The active implementation lives in `scar/`, with runnable entry
points in `scripts/` and paper-style configs in `configs/`.

## Layout

```text
SCAR/
├── scar/                         # SCAR models, objectives, data loading, eval, runtime
├── scripts/                      # train/eval/inference launch points
├── configs/                      # curated Robotwin configs for paper experiments
├── tools/data/                   # RoboTwin/LIBERO conversion and split utilities
├── data/                         # expected dataset root, ignored by git
└── outputs/                      # checkpoints/eval outputs, ignored by git
```

## Setup

Install the package from the repository root:

```bash
cd /path/to/SCAR
pip install -e .
```

SCAR uses the Wan/FDM implementation from `large-video-planner` as an external
dependency. Point `SCAR_LVP_ROOT` at that checkout if it is not next to this
repo:

```bash
export SCAR_LVP_ROOT=/path/to/large-video-planner
```

If `SCAR_LVP_ROOT` is not set, SCAR looks for a sibling
`large-video-planner` checkout next to this repository.

## Main Training

Example: SCAR KL+GRL on the Franka low-data Robotwin setting:

```bash
bash scripts/launch_train.sh \
  robotwin_adaptation/place_a2b_left/place_a2b_left__shared_latent__target_franka__m10__target_val__grl_kl.yaml
```

Useful configs:

- `place_a2b_left__target_only_gt__target_franka__m10__target_val.yaml`
- `place_a2b_left__shared_gt__target_franka__m10__target_val.yaml`
- `place_a2b_left__target_only_latent__target_franka__m10__target_val.yaml`
- `place_a2b_left__shared_latent__target_franka__m10__target_val.yaml`
- `place_a2b_left__shared_latent__target_franka__m10__target_val__kl_5e4.yaml`
- `place_a2b_left__shared_latent__target_franka__m10__target_val__grl.yaml`
- `place_a2b_left__shared_latent__target_franka__m10__target_val__grl_kl.yaml`

## Evaluation

```bash
bash scripts/launch_eval.sh \
  robotwin_adaptation/place_a2b_left/place_a2b_left__shared_latent__target_franka__m10__target_val__grl_kl.yaml \
  /path/to/checkpoints/step_0010000.pt
```

Repeated eval with aggregation:

```bash
bash scripts/launch_eval_repeated.sh \
  robotwin_adaptation/place_a2b_left/place_a2b_left__shared_latent__target_franka__m10__target_val__grl_kl.yaml \
  /path/to/checkpoints/step_0010000.pt \
  outputs/eval_repeat --num-runs 3
```

## Action-To-Latent Controller

Train the sequence-level A2L controller from a trained SCAR checkpoint:

```bash
python scripts/train_action_controller.py \
  --ckpt /path/to/scar/checkpoints/step_0010000.pt \
  --train-split train --train-dataset-filter franka \
  --eval-split eval --eval-dataset-filter franka \
  --train-windows-per-dataset 0 --eval-windows-per-dataset 0 \
  --context-len 17 \
  --controller-architecture latent_residual_cross_attention
```

Evaluate a trained controller:

```bash
bash scripts/launch_controller_eval.sh /path/to/latent_action_controller_best_eval.pt
```

Joint controller+FDM adaptation for the Sequence-A2L-FT setting:

```bash
bash scripts/launch_controller_fdm_finetune.sh \
  robotwin_adaptation/place_a2b_left/place_a2b_left__shared_latent__target_franka__m50__target_val__grl_kl__controller_fdm_joint_low_noise_weighted_cotrain_idm_teacher.yaml \
  /path/to/latent_action_controller_best_eval.pt
```

## Data Utilities

RoboTwin conversion and split generation live under `tools/data/`.

```bash
python tools/data/convert_robotwin_hf_task_family.py \
  --task-family-root /path/to/RoboTwin/hf_dataset/place_a2b_left \
  --output-root data

python tools/data/build_robotwin_adaptation_splits.py \
  --conversion-manifest data/datasets/robotwin_lmdb/manifests/robotwin_place_a2b_left_randomized_500_conversion.json
```

Generated configs should point to paths under `data/` and write run artifacts
under `outputs/`.
