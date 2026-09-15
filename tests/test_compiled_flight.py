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
    assert ball_physics._rk4_dispatch is None


def test_the_entry_point_works_without_compilation(flight):
    """Sans compilation, `_rk4` doit se comporter exactement comme `rk4_step`."""

    enable_compiled_flight(False)
    position, velocity, spin = flight
    direct = ball_physics.rk4_step(position, velocity, spin, 0.01, DEFAULT_BALL_PHYSICS)
    through = ball_physics._rk4(position, velocity, spin, 0.01, DEFAULT_BALL_PHYSICS)
    for expected, got in zip(direct, through, strict=True):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
