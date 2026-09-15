"""Le tirage des conditions de lancer ne doit plus vivre dans le chemin chaud.

Les deux rejets d'origine bouclaient avec un `.all().item()` par tour — jusqu'à
28 synchronisations GPU vers hôte par reset. Et un reset a lieu à *chaque* pas
de contrôle : avec 2048 environnements et des épisodes de ~69 pas, une
trentaine d'environnements se terminent par pas. Mesuré sur le run du
14 septembre 2026 : 99,3 % du temps en collecte, 27 % d'utilisation GPU, et un
coût de reset quasi indépendant du nombre d'environnements concernés (0,46 s
pour 1, 0,75 s pour 512) — signature d'un coût de latence, pas de calcul.

Le tirage d'un lancer ne dépend que du niveau de curriculum, jamais de la
politique : il est donc fait une fois par palier, et les resets se servent par
indexation. Ces tests verrouillent le déclenchement des reconstructions — un
vivier reconstruit trop souvent annulerait tout le bénéfice, un vivier jamais
reconstruit figerait le curriculum.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tabletennis_env import TableTennisWarpEnv, tabletennis_p2_cfg  # noqa: E402
from train_multigpu import apply_environment_config  # noqa: E402


def _env(**overrides):
    """Environnement squelette : le constructeur complet compile MuJoCo Warp."""

    env = object.__new__(TableTennisWarpEnv)
    env.cfg = tabletennis_p2_cfg()
    for key, value in overrides.items():
        setattr(env.cfg, key, value)
    env.num_envs = 8
    env.device = torch.device("cpu")
    env._launch_pool = None
    env._launch_pool_bucket = None
    env._launch_pool_draws = 0
    env.builds = 0

    def fake_sample(n_candidates, scale):
        env.builds += 1
        env.last_scale = scale
        n = max(1, n_candidates // 2)  # moitié rejetée, comme en vrai
        return (
            torch.full((n, 3), float(scale)),
            torch.ones(n, 3),
            torch.zeros(n, 3),
        )

    env._sample_valid_launches = fake_sample
    return env


def test_pool_is_built_once_and_then_only_indexed():
    """Le coût du tirage doit sortir du chemin chaud : un build, puis rien."""

    env = _env()
    env._draw_launches(8, 1.0)
    assert env.builds == 1
    for _ in range(50):
        env._draw_launches(8, 1.0)
    assert env.builds == 1, "le vivier a été reconstruit dans le chemin chaud"


def test_pool_follows_the_curriculum():
    """Un vivier figé servirait des balles trop faciles jusqu'à la fin."""

    env = _env()
    env._draw_launches(8, 0.25)
    first = env.builds
    env._draw_launches(8, 0.30)  # un palier plus haut (pas de 0.05)
    assert env.builds == first + 1
    # La valeur servie suit bien le nouveau palier.
    qpos, _, _ = env._draw_launches(8, 0.30)
    assert qpos[0, 0] == pytest.approx(0.30, abs=1e-6)


def test_a_curriculum_nudge_within_one_step_does_not_rebuild():
    """Sans quantification, le vivier serait reconstruit à chaque itération."""

    env = _env()
    env._draw_launches(8, 0.50)
    before = env.builds
    env._draw_launches(8, 0.51)
    env._draw_launches(8, 0.52)
    assert env.builds == before


def test_pool_is_renewed_so_launches_do_not_repeat_forever():
    env = _env(launch_pool_refresh=100)
    env._draw_launches(8, 1.0)
    before = env.builds
    for _ in range(20):  # 160 tirages > 100
        env._draw_launches(8, 1.0)
    assert env.builds > before


def test_draw_returns_the_requested_count_and_is_independent():
    env = _env()
    qpos, qvel, spin = env._draw_launches(5, 1.0)
    assert qpos.shape == (5, 3) and qvel.shape == (5, 3) and spin.shape == (5, 3)
    # Les tirages sont des copies : écrire dans un reset ne doit pas corrompre
    # le vivier pour les suivants.
    qpos[:] = 999.0
    again, _, _ = env._draw_launches(5, 1.0)
    assert (again != 999.0).all()


def test_yaml_exposes_the_environment_count_and_sampling_knobs():
    """`num_envs` était codé en dur : une ablation ne pouvait pas le toucher."""

    config = yaml.safe_load((REPO_ROOT / "default_config.yaml").read_text(encoding="utf-8"))
    env_cfg = apply_environment_config(tabletennis_p2_cfg(), config)
    assert env_cfg.num_envs == config["num_envs"]
    assert env_cfg.plan_candidates == config["sampling"]["plan_candidates"]
    assert env_cfg.launch_pool_size == config["sampling"]["launch_pool_size"]
