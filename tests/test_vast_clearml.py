"""Regressions for the ClearML tracking helpers and the job-queue runner.

These stay pure-Python: no GPU, no network, no ClearML server. What they pin is
the behaviour that is easy to get wrong and expensive to discover on a rented
machine — a queue file that only fails once billing has started, or a tracking
helper that raises and takes a training run down with it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from clearml_tracking import (  # noqa: E402
    CheckpointUploader,
    _default_tags,
    close_tracking,
    init_tracking,
    report_run_cost,
)
from run_jobs import JobError, build_command, build_config, load_jobs  # noqa: E402


class _StubTask:
    """Task ClearML minimale : enregistre les appels au lieu de les envoyer."""

    def __init__(self, fail: bool = False):
        self.uploaded: list[str] = []
        self.values: dict[str, float] = {}
        self.closed = False
        self.fail = fail

    def upload_artifact(self, name, artifact_object, **_):
        if self.fail:
            raise RuntimeError("serveur injoignable")
        self.uploaded.append(str(artifact_object))

    def get_logger(self):
        task = self

        class _Logger:
            def report_single_value(self, name, value):
                task.values[name] = value

        return _Logger()

    def flush(self, **_):
        pass

    def close(self):
        self.closed = True


# --------------------------------------------------------------------------
# clearml_tracking
# --------------------------------------------------------------------------


def test_tracking_is_opt_in_and_main_process_only():
    """Un run sans tracking, ou un rang secondaire, ne crée aucune Task.

    En multi-GPU, laisser chaque rang créer sa Task produirait plusieurs runs
    concurrents écrivant les mêmes courbes sous le même nom.
    """

    assert init_tracking({"tracking": {"enabled": False}}, "logs/run") is None
    assert init_tracking({}, "logs/run") is None
    assert init_tracking({"tracking": {"enabled": True}}, "logs/run", rank=1) is None


def test_default_tags_describe_the_experiment():
    tags = _default_tags(
        {
            "action_type": "muscle_pd",
            "spin": {"enabled": True, "command_mode": "uniform", "magnus_enabled": False},
        }
    )
    assert "action:muscle_pd" in tags
    assert "spin:uniform" in tags
    assert "ablation:no-magnus" in tags

    assert "spin:off" in _default_tags({"action_type": "joint_pd", "spin": {"enabled": False}})


def test_uploader_sends_each_file_once(tmp_path):
    task = _StubTask()
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "100.mp4").write_bytes(b"x")
    uploader = CheckpointUploader(task, str(tmp_path), upload_videos=True)

    uploader.flush()
    assert len(task.uploaded) == 1

    # Un second flush ne renvoie rien : sur un run long, réenvoyer chaque
    # vidéo à chaque tour saturerait la bande passante de l'instance.
    uploader.flush()
    assert len(task.uploaded) == 1

    (tmp_path / "videos" / "200.mp4").write_bytes(b"x")
    uploader.flush()
    assert len(task.uploaded) == 2


def test_upload_failure_is_swallowed_and_not_retried(tmp_path):
    """Perdre un envoi ne doit ni tuer le run, ni boucler indéfiniment."""

    task = _StubTask(fail=True)
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "100.mp4").write_bytes(b"x")
    uploader = CheckpointUploader(task, str(tmp_path))

    uploader.flush()  # ne lève pas
    assert uploader._pending() == []


def test_cost_is_reported_only_when_the_hourly_price_is_known(monkeypatch):
    task = _StubTask()
    monkeypatch.delenv("VAST_DPH", raising=False)
    report_run_cost(task, started_at=0.0)
    assert "train_hours" in task.values
    assert "train_cost_usd" not in task.values

    monkeypatch.setenv("VAST_DPH", "0.5")
    report_run_cost(task, started_at=0.0)
    assert task.values["train_cost_usd"] == pytest.approx(task.values["train_hours"] * 0.5, rel=1e-6)


def test_closing_without_a_task_is_a_no_op():
    close_tracking(None, None)  # ne lève pas


# --------------------------------------------------------------------------
# run_jobs
# --------------------------------------------------------------------------


def _write(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "jobs.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_shipped_example_queue_is_valid():
    """Le fichier d'exemple doit rester exécutable : c'est le point d'entrée."""

    jobs = load_jobs(REPO_ROOT / "jobs" / "example.yaml")
    assert jobs
    for job in jobs:
        build_config(job, base_path=REPO_ROOT / "default_config.yaml")


@pytest.mark.parametrize(
    ("document", "fragment"),
    [
        ({"runs": []}, "jobs"),
        ({"jobs": []}, "non vide"),
        ({"jobs": [{"nom": "x"}]}, "inconnue"),
        ({"jobs": [{"args": {"epochs": 3}}]}, "inconnu"),
    ],
)
def test_malformed_queues_are_rejected(tmp_path, document, fragment):
    """La validation tourne AVANT de louer une machine (vast.sh --dry-run)."""

    with pytest.raises(JobError, match=fragment):
        load_jobs(_write(tmp_path, document))


def test_override_of_an_unknown_key_is_rejected():
    """Une ablation qui ne s'applique pas produirait un résultat trompeur."""

    with pytest.raises(JobError, match="clé inconnue"):
        build_config(
            {"overrides": {"spin.magnus_enabeld": False}},
            base_path=REPO_ROOT / "default_config.yaml",
        )
    with pytest.raises(JobError, match="pas une section"):
        build_config(
            {"overrides": {"algorithm.gamma.x": 1}},
            base_path=REPO_ROOT / "default_config.yaml",
        )


def test_cli_argument_placed_in_overrides_gets_a_pointed_error():
    """`seed` et `action_type` ne sont pas dans le YAML : dire où les mettre."""

    with pytest.raises(JobError, match="placez-le sous 'args'"):
        build_config(
            {"overrides": {"action_type": "muscle_pd"}},
            base_path=REPO_ROOT / "default_config.yaml",
        )


def test_dotted_overrides_reach_nested_config():
    config = build_config(
        {"overrides": {"spin.magnus_enabled": False, "max_iterations": 7}},
        base_path=REPO_ROOT / "default_config.yaml",
    )
    assert config["spin"]["magnus_enabled"] is False
    assert config["max_iterations"] == 7
    # La config de base n'est pas modifiée pour les jobs suivants.
    base = yaml.safe_load((REPO_ROOT / "default_config.yaml").read_text(encoding="utf-8"))
    assert base["spin"]["magnus_enabled"] is True


def test_command_uses_all_gpus_by_default():
    """Une carte inutilisée sur une instance louée se paie au prix fort."""

    command = build_command({}, "/tmp/c.yaml", index=1)
    assert command[-2:] == ["--gpu-ids", "all"]
    assert "--config" in command

    command = build_command({"args": {"gpu_ids": "0 1", "seed": 3}}, "/tmp/c.yaml", index=1)
    assert command[-3:] == ["--gpu-ids", "0", "1"]
    assert "--seed" in command and "3" in command


# --------------------------------------------------------------------------
# packaging
# --------------------------------------------------------------------------


def _declared_distributions() -> set[str]:
    import re

    import tomllib

    raw = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    names = {
        re.split(r"[<>=!~\[;]", item)[0].strip().lower().replace("-", "_")
        for item in raw["project"]["dependencies"]
    }
    # Quelques distributions n'exposent pas un module du même nom.
    alias = {
        "rsl_rl_lib": "rsl_rl",
        "warp_lang": "warp",
        "pyyaml": "yaml",
        "opencv_python": "cv2",
        "pillow": "PIL",
    }
    return names | {alias[name] for name in names if name in alias}


def _third_party_imports() -> dict[str, set[str]]:
    import ast

    first_party = {path.stem for path in REPO_ROOT.glob("*.py")}
    first_party |= {"mjlab", "scripts", "tests"}
    stdlib = set(sys.stdlib_module_names)

    found: dict[str, set[str]] = {}
    sources = list(REPO_ROOT.glob("*.py")) + list((REPO_ROOT / "scripts").glob("*.py"))
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module.split(".")[0]]
            else:
                continue
            for module in modules:
                if module in stdlib or module in first_party:
                    continue
                found.setdefault(module, set()).add(path.name)
    return found


def test_every_direct_import_is_a_declared_dependency():
    """Un import direct doit être une dépendance déclarée, pas un transitif.

    `ml_collections` manquait à `pyproject.toml` tout en étant importé par
    `tabletennis_env`: présent dans le venv de développement pour l'avoir
    installé à la main un jour, absent partout ailleurs. Le défaut n'est
    apparu qu'en louant un GPU, le conteneur mourant sur un
    ModuleNotFoundError. Les imports qui arrivent par le transitif sont le
    même piège en sursis: le jour où un paquet tiers cesse de les tirer, le
    run casse loin d'ici.
    """

    declared = _declared_distributions()
    undeclared = {
        module: sorted(files)
        for module, files in _third_party_imports().items()
        if module not in declared
    }
    assert not undeclared, (
        "imports non déclarés dans pyproject.toml : "
        + ", ".join(f"{module} ({', '.join(files)})" for module, files in sorted(undeclared.items()))
    )
