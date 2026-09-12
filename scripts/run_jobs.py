#!/usr/bin/env python
"""Exécute une file de runs d'entraînement décrite en YAML.

Motivation : ``train_multigpu.py`` n'expose que cinq arguments (``--seed``,
``--action-type``, ``--learning-rate``, ``--gpu-ids``, ``--log-dir``), alors
qu'une campagne d'ablation doit faire varier n'importe quelle clé de la config
— ``spin.magnus_enabled``, ``spin.command_mode``, ``spin.target``… Plutôt que
d'étendre l'argparse indéfiniment, chaque job écrit **sa propre config
dérivée** (config de base + surcharges pointées) dans un fichier temporaire,
passé via ``--config``. Toute la surface de configuration devient atteignable
sans toucher au point d'entrée.

Chaque job tourne dans un sous-processus : un run qui plante — OOM, GPU perdu —
n'emporte pas la file avec lui.

    python scripts/run_jobs.py jobs/example.yaml --dry-run   # valider
    python scripts/run_jobs.py jobs/example.yaml             # exécuter
    ./scripts/vast.sh queue <offer_id> jobs/example.yaml     # sur vast.ai
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "default_config.yaml"

# Arguments acceptés par train_multigpu.py. Une clé inconnue dans un job est
# refusée à la validation, pas découverte sur une machine déjà payée.
_JOB_KEYS = {"name", "config", "base_config", "overrides", "args", "skip"}
_CLI_KEYS = {"seed", "action_type", "learning_rate", "gpu_ids", "log_dir"}


class JobError(ValueError):
    """Fichier de jobs mal formé — levé avant toute exécution."""


def _set_dotted(config: dict, dotted: str, value: Any) -> None:
    """Applique ``a.b.c: valeur`` dans un dict imbriqué.

    Un chemin qui ne mène nulle part est une erreur : une ablation silencieuse
    qui ne s'applique pas produit un résultat qu'on croit informatif et qui ne
    l'est pas. Mieux vaut refuser le fichier.
    """

    if dotted in _CLI_KEYS:
        raise JobError(
            f"'{dotted}' est un argument de train_multigpu.py, pas une clé de config : "
            f"placez-le sous 'args' plutôt que sous 'overrides'"
        )
    parts = dotted.split(".")
    node = config
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            raise JobError(f"surcharge '{dotted}' : '{part}' n'est pas une section de la config")
        node = node[part]
    if parts[-1] not in node:
        raise JobError(f"surcharge '{dotted}' : clé inconnue dans la config de base")
    node[parts[-1]] = value


def load_jobs(path: str | Path) -> list[dict]:
    """Lit et valide un fichier de jobs. Lève ``JobError`` si invalide."""

    with open(path, encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if not isinstance(document, dict) or "jobs" not in document:
        raise JobError(f"{path} : clé racine 'jobs' attendue")
    jobs = document["jobs"]
    if not isinstance(jobs, list) or not jobs:
        raise JobError(f"{path} : 'jobs' doit être une liste non vide")

    for index, job in enumerate(jobs, start=1):
        if not isinstance(job, dict):
            raise JobError(f"job #{index} : objet attendu, reçu {type(job).__name__}")
        unknown = set(job) - _JOB_KEYS
        if unknown:
            raise JobError(f"job #{index} : clé(s) inconnue(s) {sorted(unknown)}")
        args = job.get("args") or {}
        if not isinstance(args, dict):
            raise JobError(f"job #{index} : 'args' doit être un mapping")
        unknown_args = set(args) - _CLI_KEYS
        if unknown_args:
            raise JobError(
                f"job #{index} : argument(s) inconnu(s) {sorted(unknown_args)} "
                f"(acceptés : {sorted(_CLI_KEYS)})"
            )
        overrides = job.get("overrides") or job.get("config") or {}
        if not isinstance(overrides, dict):
            raise JobError(f"job #{index} : 'overrides' doit être un mapping")
    return jobs


def build_config(job: dict, *, base_path: Path) -> dict:
    """Compose la config d'un job : base + surcharges pointées."""

    path = Path(job.get("base_config") or base_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config = copy.deepcopy(config)
    for dotted, value in (job.get("overrides") or job.get("config") or {}).items():
        _set_dotted(config, dotted, value)
    return config


def build_command(job: dict, config_path: str, *, index: int) -> list[str]:
    """Ligne de commande ``train_multigpu.py`` d'un job."""

    args = job.get("args") or {}
    command = [sys.executable, str(REPO_ROOT / "train_multigpu.py"), "--config", config_path]
    if "seed" in args:
        command += ["--seed", str(args["seed"])]
    if "action_type" in args:
        command += ["--action-type", str(args["action_type"])]
    if "learning_rate" in args:
        command += ["--learning-rate", str(args["learning_rate"])]
    if "log_dir" in args:
        command += ["--log-dir", str(args["log_dir"])]
    # Par défaut toutes les cartes : sur une instance louée, laisser un GPU
    # inutilisé se paie au prix fort.
    command += ["--gpu-ids", *str(args.get("gpu_ids", "all")).split()]
    return command


def _job_name(job: dict, index: int) -> str:
    return job.get("name") or f"job{index}"


def run_jobs(path: str | Path, *, dry_run: bool = False, keep_going: bool = False) -> int:
    """Exécute la file. Renvoie le code de sortie du processus."""

    jobs = load_jobs(path)
    print(f"{len(jobs)} job(s) dans {path}\n")
    failures: list[str] = []

    for index, job in enumerate(jobs, start=1):
        name = _job_name(job, index)
        if job.get("skip"):
            print(f"[{index}/{len(jobs)}] {name} — ignoré (skip: true)")
            continue

        config = build_config(job, base_path=DEFAULT_CONFIG)
        # Le nom de la task ClearML par défaut vient du dossier de run ; ici on
        # a mieux : le nom du job, qui dit ce que l'expérience teste.
        if isinstance(config.get("tracking"), dict) and not config["tracking"].get("task_name"):
            config["tracking"]["task_name"] = name

        if dry_run:
            command = build_command(job, "<config-derivee>", index=index)
            print(f"[{index}/{len(jobs)}] {name}")
            print(f"    {' '.join(command)}")
            surcharges = job.get("overrides") or job.get("config") or {}
            if surcharges:
                print(f"    surcharges : {json.dumps(surcharges, ensure_ascii=False)}")
            continue

        # delete=False : le sous-processus doit pouvoir rouvrir le fichier, ce
        # que Windows interdit sur un handle encore ouvert. Nettoyé en finally.
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=f"_{name}.yaml", delete=False, encoding="utf-8"
        )
        try:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
            handle.close()
            command = build_command(job, handle.name, index=index)
            print(f"[{index}/{len(jobs)}] {name}")
            print(f"    {' '.join(command)}", flush=True)
            started = time.time()
            completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
            minutes = (time.time() - started) / 60.0
            if completed.returncode == 0:
                print(f"    ✓ terminé en {minutes:.1f} min\n", flush=True)
            else:
                failures.append(name)
                print(f"    ✗ échec (code {completed.returncode}) après {minutes:.1f} min\n", flush=True)
                if not keep_going:
                    break
        finally:
            os.unlink(handle.name)

    if failures:
        print(f"Jobs en échec : {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("jobs_file", help="fichier YAML décrivant la file (voir jobs/example.yaml)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="valide le fichier et affiche les commandes, sans rien exécuter",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="poursuit la file après un job en échec (défaut : s'arrêter)",
    )
    args = parser.parse_args()
    try:
        return run_jobs(args.jobs_file, dry_run=args.dry_run, keep_going=args.keep_going)
    except (JobError, FileNotFoundError) as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
