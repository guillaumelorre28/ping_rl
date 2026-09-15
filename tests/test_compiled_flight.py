"""Compiler le pas d'intégration ne doit rien changer aux résultats.

Le planificateur représentait 88 % du temps d'un pas d'entraînement sur GPU,
contre 2,8 % pour la physique MuJoCo Warp — non par calcul, mais par latence :
368 412 lancements de noyaux CUDA par pas, d'une durée moyenne de 1,3 µs, très
en dessous du coût de lancement. Ces lancements viennent du pas RK4, appelé
neuf fois par plan à travers quatre itérations de Newton, trois sous-pas et
quatre étages, chacun réévaluant un modèle aérodynamique à interpolation.

`torch.compile` les fusionne. Ce qu'il ne doit PAS faire, c'est changer la
trajectoire : ces tests comparent les deux chemins sur les mêmes entrées.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import ball_physics  # noqa: E402
from ball_physics import (  # noqa: E402
    DEFAULT_BALL_PHYSICS,
    enable_compiled_flight,
    propagate_to_height,
)


@pytest.fixture
def flight():
    torch.manual_seed(0)
    n = 32
    position = torch.tensor([1.5, 0.0, 1.0]).repeat(n, 1) + torch.randn(n, 3) * 0.05
    velocity = torch.tensor([-6.0, 0.0, 1.5]).repeat(n, 1) + torch.randn(n, 3) * 0.2
    spin = torch.randn(n, 3) * 100.0
    return position, velocity, spin


def _propagate(flight):
    position, velocity, spin = flight
    return propagate_to_height(
        position,
        velocity,
        spin,
        0.795,
        max_time=1.2,
        params=DEFAULT_BALL_PHYSICS,
        root_iterations=4,
        integration_substeps=3,
    )


def test_compiling_does_not_move_the_ball(flight):
    """Une trajectoire différente invaliderait toute comparaison de runs."""

    enable_compiled_flight(False)
    reference = _propagate(flight)

    if not enable_compiled_flight(True):
        pytest.skip("compilation indisponible sur cette plateforme")
    try:
        compiled = _propagate(flight)
    finally:
        enable_compiled_flight(False)

    names = ("instant", "position", "vitesse", "spin")
    for name, expected, got in zip(names, reference, compiled, strict=False):
        torch.testing.assert_close(
            got, expected, rtol=0, atol=1e-5, msg=f"{name} diverge après compilation"
        )
    # Le drapeau « la balle a bien croisé le plan » doit être identique, pas
    # seulement proche : c'est lui qui décide si un plan est jouable.
    assert bool((reference[4] == compiled[4]).all())


def test_disabling_restores_the_direct_path():
    """Le repli doit être total : une compilation ratée ne doit rien laisser."""

    enable_compiled_flight(True)
    assert enable_compiled_flight(False) is False
    assert ball_physics._propagate_dispatch == {}


def test_the_planner_output_stays_within_float32_noise():
    """Ce qui compte n'est pas la trajectoire brute mais la commande produite.

    La boucle de Newton du planificateur amplifie les écarts d'arrondi : la
    propagation seule dévie de 1e-7, la commande de raquette de 1e-4 en
    absolu. Rapporté aux échelles — 6,9 m/s de vitesse de raquette, 28 rad/s
    de rotation, 338 rad/s d'effet visé — cela fait au plus 1,2e-5 en relatif,
    sept ordres de grandeur sous la randomisation de domaine (±10 %). Ce que
    l'on exige en revanche à l'identique, c'est le drapeau de faisabilité :
    c'est lui qui décide si un coup est jouable.
    """

    import yaml
    from tabletennis_env import TableTennisWarpEnv, tabletennis_p2_cfg
    from train_multigpu import apply_environment_config

    config = yaml.safe_load((REPO_ROOT / "default_config.yaml").read_text(encoding="utf-8"))
    env_cfg = apply_environment_config(tabletennis_p2_cfg(), config)
    env_cfg.num_envs = 16
    env_cfg.compile_flight = False
    env = TableTennisWarpEnv(env_cfg, device="cpu")
    env.reset()
    env.curr_iter, env.total_iter = 40, 100

    position, velocity, spin = env._draw_launches(6, 1.0)
    landing = (
        torch.rand((6, 3)) * (env.opponent_table_upper - env.opponent_table_lower)
        + env.opponent_table_lower
    )
    command = (
        torch.rand((6, 3)) * (env.spin_target_high - env.spin_target_low)
        + env.spin_target_low
    )

    def plan():
        return env.get_high_command(
            position, velocity, spin, spin_command=command,
            target_landing=landing, return_valid=True,
        )

    enable_compiled_flight(False)
    reference = plan()
    if not enable_compiled_flight(True):
        pytest.skip("compilation indisponible sur cette plateforme")
    try:
        plan()  # la première passe compile
        compiled = plan()
    finally:
        enable_compiled_flight(False)

    for expected, got in zip(reference[:-1], compiled[:-1], strict=True):
        if not expected.dtype.is_floating_point:
            continue
        scale = float(expected.abs().max()) or 1.0
        assert float((expected - got).abs().max()) / scale < 1.0e-4
    assert bool((reference[-1] == compiled[-1]).all())


def test_the_entry_point_works_without_compilation(flight):
    """Sans compilation, `_rk4` doit se comporter exactement comme `rk4_step`."""

    enable_compiled_flight(False)
    position, velocity, spin = flight
    direct = ball_physics.rk4_step(position, velocity, spin, 0.01, DEFAULT_BALL_PHYSICS)
    through = ball_physics._rk4(position, velocity, spin, 0.01, DEFAULT_BALL_PHYSICS)
    for expected, got in zip(direct, through, strict=True):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
