"""Le runner vendu doit rester compatible avec la version de rsl_rl installée.

`on_policy_runner.py` est une copie modifiée du runner amont de rsl_rl 2.3.x.
Le paquet a été épinglé un temps en 3.1.0, dont la refonte « groupes
d'observations » change la signature d'`ActorCritic` :
`(obs, obs_groups, num_actions)` au lieu de
`(num_actor_obs, num_critic_obs, num_actions)`. Résultat, tout run mourait à la
construction du runner sur `TypeError: 'int' object is not subscriptable` —
défaut invisible aux tests de l'environnement, qui n'instancient jamais le
runner, et donc découvert sur une machine louée.

Ce test construit un `OnPolicyRunner` avec la vraie `default_config.yaml` et un
environnement factice. Il ne valide pas l'entraînement : seulement que les
appels du runner et l'API de rsl_rl parlent la même langue.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

NUM_ENVS = 4
NUM_OBS = 12
NUM_ACTIONS = 3


class _StubEnv:
    """Surface minimale de `VecEnv` utilisée par `OnPolicyRunner.__init__`."""

    def __init__(self):
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.device = "cpu"

    def get_observations(self):
        obs = torch.zeros(NUM_ENVS, NUM_OBS)
        return obs, {"observations": {}}


def _config() -> dict:
    config = yaml.safe_load((REPO_ROOT / "default_config.yaml").read_text(encoding="utf-8"))
    # `class_name` est retiré par le runner (`pop`) : une copie par test.
    return config


def test_runner_builds_against_the_installed_rsl_rl():
    # Import dur, pas `importorskip` : toutes les dépendances du runner sont
    # déclarées dans pyproject.toml. Un skip ici rendrait le test muet dans
    # l'environnement même où il doit parler — un venv désynchronisé.
    from on_policy_runner import OnPolicyRunner

    runner = OnPolicyRunner(
        env=_StubEnv(), eval_env=_StubEnv(), train_cfg=_config(), log_dir=None, device="cpu"
    )

    # La politique doit voir les bonnes dimensions, pas seulement se construire.
    assert runner.alg.policy.actor[0].in_features == NUM_OBS
    assert runner.alg.policy.actor[-1].out_features == NUM_ACTIONS


def test_policy_and_algorithm_accept_every_config_key():
    """Une clé silencieusement ignorée est une expérience qu'on croit régler.

    rsl_rl avale les surplus dans `**kwargs` en n'émettant qu'un `print` :
    changer `algorithm.gamma` sans effet ne lèverait rien.
    """

    import inspect

    from rsl_rl.algorithms import PPO
    from rsl_rl.modules import ActorCritic

    config = _config()
    for section, cls in (("policy", ActorCritic), ("algorithm", PPO)):
        accepted = set(inspect.signature(cls.__init__).parameters)
        unknown = {key for key in config[section] if key != "class_name"} - accepted
        assert not unknown, f"{section}: clés ignorées par {cls.__name__} : {sorted(unknown)}"


def test_linear_decay_schedule_lives_in_the_vendored_runner():
    """`schedule: linear_decay` n'existe dans aucune version de rsl_rl.

    C'est le runner local qui l'implémente. Le jour où quelqu'un remplace ce
    fichier par le runner amont, le planning retomberait silencieusement sur un
    taux d'apprentissage constant.
    """

    config = _config()
    if config["algorithm"].get("schedule") != "linear_decay":
        pytest.skip("la config n'utilise pas linear_decay")
    source = (REPO_ROOT / "on_policy_runner.py").read_text(encoding="utf-8")
    assert "linear_decay" in source
