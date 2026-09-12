# Diff-Muscle

**Diff-Muscle: Efficient Learning for Musculoskeletal Robotic Table Tennis. [[Paper](https://arxiv.org/abs/2603.08617)]**

<p align="center">
  <img width="800" height="400" alt="image" src="images\diff-muscle.png" />
  <p align="center"><i>Overview of Diff-Muscle</i></p>
</p>

---

## Overview

This repository is the official implementation of our paper [Diff-muscle](https://arxiv.org/abs/2603.08617) and the First Place solution for 2025 NeurIPS - MyoChallenge: Towards Human Athletic Intelligence Table Tennis Track - Team ActingAI. We use [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) for GPU-accelerated training, supporting thousands of parallel environments.



---

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and requires an **NVIDIA GPU** with CUDA 12.4+.

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repository
git clone <your-repo-url>
cd Diff-Muscle

# 3. Install all dependencies (includes mjlab and mujoco-warp)
uv sync
source ./.venv/bin/activate
```

This project bundles [mjlab](https://github.com/mujocolab/mjlab) in src/mjlab/. Running uv sync installs mjlab from this local source along with all its dependencies, including [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) (pinned to a tested revision). No separate mjlab installation is required.

---

## Training

### Single GPU

```bash
python train_multigpu.py
```

### Multi-GPU

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" python train_multigpu.py --gpu-ids all
```

Training configuration is in `default_config.yaml`. Key parameters:

**Environment**


| Parameter            | Default    | Description                            |
| -------------------- | ---------- | -------------------------------------- |
| `num_envs`           | 1024       | Number of parallel environments        |
| `action_type`        | `joint_pd` | Action space: `joint_pd`, `muscle_act` |
| `max_episode_length` | 300        | Max steps per episode                  |
| `frame_skip`         | 5          | Physics steps per control step         |

### Learning ball spin

The ball model includes aerodynamic drag, the Magnus force, and coupled
linear/angular impacts with the table and racket. Spin commands are configured
under `spin` in `default_config.yaml` and use trajectory coordinates in rad/s:

```yaml
spin:
  command_mode: fixed
  target: [300.0, 0.0, 0.0]  # topspin
  reward_axis_weights: [1.0, 0.0, 0.0]  # optimize topspin only
  reward_weight: 60.0
```

Use `[-300.0, 0.0, 0.0]` for backspin, `[0.0, 200.0, 0.0]` for sidespin, or
`[0.0, 0.0, 0.0]` for a no-spin return. `command_mode: uniform` samples from
`target_range` and trains a single command-conditioned policy. The spin reward
is gated by a legal return, so increasing `reward_weight` cannot reward a miss.
`reward_sigma` controls the tolerance independently for topspin, sidespin, and
corkscrew, while `reward_axis_weights` selects which components contribute to
the objective. The default uniform range keeps corkscrew at zero: with an isotropic
single-point contact it is not independently controllable, unlike topspin,
backspin, and sidespin. A non-zero third component remains available for model
identification experiments with a richer contact model.

The optional `spin.curriculum` scales both incoming and requested spin from
`initial_scale` to full amplitude by `full_fraction` of training. Evaluation
always uses the full configured ranges.

The repository default is `command_mode: uniform`, which learns one policy
conditioned on topspin/backspin and sidespin commands. Switch to `fixed` as in
the example above when training one effect at a time. Planner-imitation rewards
are annealed with `spin.shaping`, while the legal-return, placement, and spin
objectives keep their full weight.

For a strict pre-spin ablation, set `enabled: false`; this also restores the old
observation/reward dimensions and paddle target convention. Independent
ablations are available through `drag_enabled`, `magnus_enabled`,
`incoming_spin_enabled`, `planner_enabled`, `reward_enabled`, and
`analytic_contact_override`. MultiCCD stays disabled because it is not
implemented by the pinned MuJoCo-Warp revision.

See [`docs/spin_physics.md`](docs/spin_physics.md) for the equations, planner
design, limitations, and calibration protocol.


**Runner**


| Parameter                 | Default | Description                                     |
| ------------------------- | ------- | ----------------------------------------------- |
| `num_steps_per_env`       | 20      | Rollout length per environment per update       |
| `max_iterations`          | 3000    | Total number of policy update steps             |
| `empirical_normalization` | `true`  | Normalize observations using running statistics |
| `eval_interval`           | 500     | Run evaluation every N iterations               |
| `save_interval`           | 300     | Save checkpoint every N iterations              |
| `eval_episodes`           | 20      | number of episodes to evaluate                  |


**PPO Algorithm**


| Parameter             | Default        | Description                                                                |
| --------------------- | -------------- | -------------------------------------------------------------------------- |
| `learning_rate`       | 0.0005         | Initial learning rate (decays linearly by default)                         |
| `schedule`            | `linear_decay` | LR schedule: `linear_decay` or `adaptive`                                  |
| `num_learning_epochs` | 5              | Gradient update epochs per rollout                                         |
| `num_mini_batches`    | 4              | Mini-batches per epoch (`batch = num_envs × num_steps / num_mini_batches`) |


---

## Project Structure

```
Diff-Muscle/
├── tabletennis_env.py        # Main RL environment (TableTennisWarpEnv)
├── planner.py                # Physics-based ball trajectory planner
├── muscle_utils.py           # Muscle activation utilities (PD + FLV inverse dynamics)
├── on_policy_runner.py       # PPO on-policy training runner
├── train_multigpu.py  # Training entry point
├── default_config.yaml# PPO hyperparameters and logging config
├── tabletennis.xml           # MuJoCo scene definition
├── assets/                   # 3D mesh assets (paddle, ball, table)
├── myo_sim/                  # MyoSim musculoskeletal models (Apache-2.0)
├── src/mjlab/                # mjlab framework source
└── tests/                    # Unit tests
```

## Third-Party Code

- `**src/mjlab/**` — [mjlab](https://github.com/mujocolab/mjlab) framework (Apache-2.0)
- `**src/mjlab/utils/lab_api/**` — Utilities forked from [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause)

---

## Citation

If you find this open source release useful, please reference in your paper:

```
@article{zhao2026diff,
  title={Diff-Muscle: Efficient Learning for Musculoskeletal Robotic Table Tennis},
  author={Zhao, Wentao and Guo, Jun and Huang, Kangyao and Liu, Xin and Liu, Huaping},
  journal={arXiv preprint arXiv:2603.08617},
  year={2026}
}
```
