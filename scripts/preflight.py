#!/usr/bin/env python
"""Vérification avant location : monter toute la pile, sans entraîner.

Trois défauts ont été découverts sur une machine louée plutôt qu'ici, chacun
coûtant une location et un aller-retour : une dépendance non déclarée, un
drapeau MuJoCo renommé, et un `OnPolicyRunner` incompatible avec la version
de rsl_rl épinglée. Les deux premiers auraient été vus par un simple import ;
le troisième non — la suite de tests n'instancie jamais le runner sur le vrai
environnement.

Ce script fait exactement ce que fait `train_multigpu.run_train`, jusqu'à la
construction du runner et un pas d'entraînement, puis s'arrête. Il tourne sur
CPU avec une poignée d'environnements, donc en quelques minutes et sans GPU :

    python scripts/preflight.py                       # dans le venv local
    docker run --rm <image> python scripts/preflight.py

Il ne remplace pas la suite de tests : il la complète là où elle est aveugle,
c'est-à-dire sur l'assemblage réel des pièces.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--action-type",
        default="both",
        choices=["joint_pd", "muscle_pd", "both"],
        help="muscle_pd est l'espace d'action du papier ; il emprunte un chemin distinct",
    )
    args = parser.parse_args()

    # `OnPolicyRunner.learn` appelle `store_code_state`, qui lance `git diff`.
    # Une config git globale avec `diff.external` (ici un script maison
    # appelant opendiff, qui exige Xcode) fait mourir l'appel — panne de poste
    # de travail, sans rapport avec ce qu'on vérifie. On neutralise la config
    # GLOBALE pour ce processus seulement ; la config du dépôt est conservée.
    os.environ.setdefault("GIT_CONFIG_GLOBAL", os.devnull)

    import torch
    import yaml

    from on_policy_runner import OnPolicyRunner
    from tabletennis_env import TableTennisWarpEnv, tabletennis_p2_cfg
    from train_multigpu import apply_environment_config

    config = yaml.safe_load((REPO_ROOT / "default_config.yaml").read_text(encoding="utf-8"))
    # Le suivi ClearML n'a rien à faire ici : on vérifie l'assemblage, et une
    # Task créée à chaque préflight polluerait le projet.
    config.setdefault("tracking", {})["enabled"] = False

    types = ["joint_pd", "muscle_pd"] if args.action_type == "both" else [args.action_type]
    for action_type in types:
        env_cfg = apply_environment_config(tabletennis_p2_cfg(), config)
        env_cfg.num_envs = args.num_envs
        env_cfg.action_type = action_type
        env = TableTennisWarpEnv(env_cfg, device=args.device)
        obs, _ = env.reset()
        print(f"{action_type:10s} env      obs={tuple(obs.shape)} actions={env.num_actions}")

        eval_cfg = apply_environment_config(tabletennis_p2_cfg(), config)
        eval_cfg.eval_env = True
        eval_cfg.num_envs = 1
        eval_cfg.action_type = action_type
        eval_cfg.enable_domain_randomization = False
        eval_cfg.enable_action_randomization = False
        eval_env = TableTennisWarpEnv(eval_cfg, device=args.device)
        eval_env.reset()

        # C'est ICI que le défaut rsl_rl 3.x tuait chaque run.
        with tempfile.TemporaryDirectory() as log_dir:
            runner = OnPolicyRunner(
                env=env,
                eval_env=eval_env,
                # deepcopy : le runner fait `pop("class_name")` sur
                # `config['algorithm']` et `config['policy']`. Une copie de
                # surface les partagerait, et le second type d'action
                # mourrait sur KeyError.
                train_cfg=copy.deepcopy(config),
                log_dir=log_dir,
                device=args.device,
            )
            print(f"{action_type:10s} runner   {type(runner.alg).__name__} construit")

            # Une itération complète : collecte + mise à jour. Un runner qui se
            # construit mais ne peut pas apprendre n'aurait rien prouvé.
            runner.learn(num_learning_iterations=1)

        params = torch.cat([p.flatten() for p in runner.alg.policy.parameters()])
        finite = bool(torch.isfinite(params).all())
        print(f"{action_type:10s} learn    1 itération OK, poids finis={finite}")
        if not finite:
            print("✗ des poids non finis après une itération", file=sys.stderr)
            return 1

    print("\n✓ préflight complet : environnement, runner et une itération d'entraînement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
