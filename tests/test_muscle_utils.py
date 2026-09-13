"""Regressions for the PD -> muscle-activation conversion.

The conversion mixes tensors with two different leading dimensions: data is
per-environment, while model fields are shared (leading dimension 1) unless
explicitly expanded. Getting that wrong does not raise -- it silently applies
one environment's coefficients to all of them -- so the checks below are about
numerical independence, not just shapes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muscle_utils import align_model_field, target_length_to_activations  # noqa: E402

NUM_ACTUATORS = 6
NUM_MUSCLES = 4


def _fake_model_and_data(num_envs: int):
    """A model whose fields are shared, and per-environment data."""

    dyntype = torch.zeros(NUM_ACTUATORS, dtype=torch.int32)
    dyntype[:NUM_MUSCLES] = 4  # mjDYN_MUSCLE

    lengthrange = torch.zeros(1, NUM_ACTUATORS, 2)
    lengthrange[..., 0] = 0.4
    lengthrange[..., 1] = 0.9

    # The muscle defaults this model actually uses, from tabletennis.xml:
    # range, force, scale, lmin, lmax, vmax, fpmax, fvmax. Values outside the
    # active band flatten the force-length curve and would make the checks
    # below vacuous, so keep them realistic.
    prm = torch.tensor([0.75, 1.05, 200.0, 200.0, 0.5, 1.6, 1.5, 1.3, 1.2, 0.0])
    biasprm = prm.view(1, 1, 10).repeat(1, NUM_ACTUATORS, 1)

    model = SimpleNamespace(
        actuator_dyntype=dyntype,
        actuator_lengthrange=lengthrange,
        actuator_biasprm=biasprm,
        actuator_gainprm=biasprm.clone(),
        actuator_acc0=torch.full((1, NUM_ACTUATORS), 1.0),
    )
    data = SimpleNamespace(
        actuator_length=torch.full((num_envs, NUM_ACTUATORS), 0.55),
        actuator_velocity=torch.zeros(num_envs, NUM_ACTUATORS),
    )
    return model, data


def test_align_model_field_separates_shared_from_expanded():
    shared = torch.zeros(1, 5, 2)
    expanded = torch.zeros(3, 5, 2)

    assert align_model_field(shared, 3).shape == (3, 5, 2)
    assert align_model_field(expanded, 3) is expanded
    # Une dimension de tête inattendue doit échouer bruyamment plutôt que de
    # produire un broadcast silencieux.
    with pytest.raises(ValueError, match="leading dimension"):
        align_model_field(torch.zeros(2, 5, 2), 3)


def test_each_environment_gets_its_own_activation():
    """Le piège : réutiliser les coefficients de l'env 0 pour tous les autres.

    Les champs de modèle ont une dimension de tête de 1, les données une par
    environnement. Dériver la taille de lot des premiers réduisait les
    lancements Warp à un seul environnement, dont bias et gain se rediffusaient
    sur les autres par broadcast — sans lever la moindre erreur.
    """

    num_envs = 4
    model, data = _fake_model_and_data(num_envs)
    # Une longueur mesurée différente par environnement : gain et activation
    # doivent suivre, puisqu'ils sortent des noyaux Warp état par état.
    data.actuator_length += torch.linspace(0.0, 0.2, num_envs).unsqueeze(-1)
    target = torch.full((num_envs, NUM_ACTUATORS), 0.55)

    activations, bias, gain, force = target_length_to_activations(
        model, data, target, kp_scale=1.0, kd_scale=1.0
    )

    assert activations.shape == (num_envs, NUM_MUSCLES)
    assert torch.isfinite(activations).all()
    assert ((activations >= 0.0) & (activations <= 1.0)).all()

    assert len(torch.unique(gain[:, 0])) == num_envs, "les environnements partagent un gain"
    assert len(torch.unique(activations[:, 0])) == num_envs, "activations rediffusées"
    # L'env 0 est déjà à la longueur visée : rien à tirer.
    assert activations[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert torch.all(activations[1:, 0] > activations[:-1, 0])


def test_expanded_model_field_is_used_per_environment():
    """Une force maximale randomisée par env doit produire des forces distinctes."""

    num_envs = 3
    model, data = _fake_model_and_data(num_envs)
    biasprm = model.actuator_biasprm.repeat(num_envs, 1, 1)
    biasprm[:, :, 2] = torch.tensor([100.0, 200.0, 400.0]).unsqueeze(-1)
    model.actuator_biasprm = biasprm

    target = torch.full((num_envs, NUM_ACTUATORS), 0.5)
    _, _, _, force = target_length_to_activations(
        model, data, target, kp_scale=10.0, kd_scale=1.0
    )

    ratios = force[:, 0] / force[0, 0]
    assert ratios[1] == pytest.approx(2.0, rel=1e-4)
    assert ratios[2] == pytest.approx(4.0, rel=1e-4)
