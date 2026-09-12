"""Configuration and curriculum regressions for spin learning."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from planner import DEFAULT_PADDLE_HALF_THICKNESS, PAD_CENTER_OFFSET_LOCAL
from tabletennis_env import (
    TableTennisWarpEnv,
    apply_spin_ablation_config,
    tabletennis_p2_cfg,
)
from train_multigpu import apply_environment_config


def test_default_yaml_maps_to_spin_environment_config():
    config_path = Path(__file__).resolve().parents[1] / "default_config.yaml"
    with config_path.open(encoding="utf-8") as stream:
        training_config = yaml.safe_load(stream)

    env_config = apply_environment_config(tabletennis_p2_cfg(), training_config)

    assert env_config.spin_physics_enabled
    assert env_config.spin_command_mode == "uniform"
    assert env_config.aerodynamic_model == "conti2026"
    assert not env_config.enable_multiccd
    assert env_config.spin_planner_root_iterations == 4
    assert env_config.spin_planner_integration_substeps == 3
    assert tuple(env_config.spin_reward_axis_weights) == (1.0, 1.0, 0.0)
    assert env_config.weighted_reward_keys["spin_target"] == pytest.approx(60.0)
    assert env_config.weighted_reward_keys["landing_target"] == pytest.approx(40.0)


def test_spin_curriculum_reaches_full_range():
    env = object.__new__(TableTennisWarpEnv)
    env.cfg = tabletennis_p2_cfg()
    env.total_iter = 100

    env.curr_iter = 0
    assert env._get_spin_curriculum_scale() == pytest.approx(0.25)

    env.curr_iter = 20
    assert env._get_spin_curriculum_scale() == pytest.approx(0.625)

    env.curr_iter = 40
    assert env._get_spin_curriculum_scale() == pytest.approx(1.0)

    env.cfg.eval_env = True
    env.curr_iter = 0
    assert env._get_spin_curriculum_scale() == pytest.approx(1.0)


def test_disabled_spin_restores_legacy_observation_and_reward_interfaces():
    cfg = tabletennis_p2_cfg()
    cfg.spin_physics_enabled = False
    cfg = apply_spin_ablation_config(cfg)

    assert "ball_spin" not in cfg.obs_keys
    assert "target_spin" not in cfg.obs_keys
    assert "ball_physics_params" not in cfg.critic_obs_keys
    assert "spin_target" not in cfg.weighted_reward_keys
    assert "landing_target" not in cfg.weighted_reward_keys
    assert "paddle_contact_vel_err" not in cfg.weighted_reward_keys


def test_contact_hysteresis_ignores_one_substep_sensor_chatter():
    env = object.__new__(TableTennisWarpEnv)
    env.num_envs = 1
    env.cfg = tabletennis_p2_cfg()
    env._contact_latched = torch.zeros(1, 6, dtype=torch.bool)
    env._contact_separation_steps = torch.zeros(1, 6, dtype=torch.int)
    env._substep_contact_info = torch.zeros(1, 6, dtype=torch.bool)
    contact = torch.tensor([[True, False, False, False, False, False]])
    clear = torch.zeros_like(contact)

    assert env._update_contact_latches(contact)[0, 0]
    env._update_contact_latches(clear)
    assert not env._update_contact_latches(contact)[0, 0]
    env._update_contact_latches(clear)
    env._update_contact_latches(clear)
    assert env._update_contact_latches(contact)[0, 0]


class _FakePaddleData:
    """Minimal stand-in exposing the qpos/qvel slices the reward term reads."""

    def __init__(self, quaternion: torch.Tensor, body_velocity: torch.Tensor, angular: torch.Tensor):
        self.qpos = torch.cat((torch.zeros(quaternion.shape[0], 3), quaternion), dim=-1)
        self.qvel = torch.cat((body_velocity, angular), dim=-1)


def _contact_velocity_env(body_velocity: torch.Tensor, angular: torch.Tensor) -> torch.Tensor:
    env = object.__new__(TableTennisWarpEnv)
    env.num_envs = body_velocity.shape[0]
    env.paddle_posadr = 0
    env.paddle_dofadr = 0
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(env.num_envs, -1)
    env.target_contact_offset_local = torch.tensor(
        [list(PAD_CENTER_OFFSET_LOCAL)]
    ).expand(env.num_envs, -1) + torch.tensor([[0.0, 0.0, -DEFAULT_PADDLE_HALF_THICKNESS]])
    # ``data`` is a read-only property forwarding to ``sim``, so stub the latter.
    env.sim = SimpleNamespace(data=_FakePaddleData(identity, body_velocity, angular))
    env._paddle_kinematics = lambda: (None, None, angular, None, None)
    return env._paddle_contact_velocity()


def test_command_reward_scores_the_contact_point_not_the_twist_split():
    """Two twists with the same contact velocity must score identically.

    The planner's split between arm translation and wrist rotation is one point
    on a continuum that leaves the ball physics unchanged, so the reward must
    not prefer one member of it.  Conversely, matching the body velocity while
    holding the wrist still is a materially different shot and must not earn the
    full term.
    """

    lever = torch.tensor([list(PAD_CENTER_OFFSET_LOCAL)]) + torch.tensor(
        [[0.0, 0.0, -DEFAULT_PADDLE_HALF_THICKNESS]]
    )
    planned_body = torch.tensor([[-3.0, 0.4, 2.0]])
    planned_angular = torch.tensor([[1.5, 12.0, -3.0]])
    target = _contact_velocity_env(planned_body, planned_angular)

    # A different split of the very same contact velocity.
    other_angular = torch.tensor([[-4.0, 5.0, 8.0]])
    other_body = target - torch.cross(other_angular, lever, dim=-1)
    equivalent = _contact_velocity_env(other_body, other_angular)
    assert torch.allclose(equivalent, target, atol=1.0e-6)

    # Tracking the body velocity with a still wrist misses omega x r entirely.
    still_wrist = _contact_velocity_env(planned_body, torch.zeros_like(planned_angular))
    missing = torch.linalg.vector_norm(still_wrist - target, dim=-1)
    assert torch.allclose(
        missing, torch.linalg.vector_norm(torch.cross(planned_angular, lever, dim=-1), dim=-1)
    )
    assert missing.item() > 0.5

    def reward(value: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * torch.linalg.vector_norm(value - target, dim=-1))

    assert reward(equivalent).item() == pytest.approx(1.0, abs=1.0e-6)
    assert reward(still_wrist).item() < 0.75
