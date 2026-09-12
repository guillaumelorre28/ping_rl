"""Suivi d'expérience ClearML pour les entraînements table tennis.

L'intégration tient en un point d'accroche : ``init_tracking`` crée la ``Task``
**avant** que ``OnPolicyRunner`` ne construise son ``SummaryWriter``. ClearML
instrumente TensorBoard à l'initialisation de la Task, donc tous les
``add_scalar`` du runner remontent ensuite d'eux-mêmes — aucune modification de
``on_policy_runner.py``, et ``logger: tensorboard`` reste le réglage à utiliser.
Choisir ``logger: wandb`` court-circuiterait cette capture.

Ce que ce module ajoute par-dessus :

* les checkpoints et les vidéos d'évaluation sont poussés **pendant** le run,
  pas seulement à la fin — sur une instance cloud détruite automatiquement, un
  run interrompu doit rester récupérable ;
* le coût de la machine est calculé et remonté comme métrique.

Règle qui vaut pour tout le fichier : **le suivi ne fait jamais échouer un
entraînement**. ClearML absent, mal configuré ou injoignable se traduit par un
avertissement et un ``None``, pas par une exception.
"""

from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

# Le runner écrit ses poids sous cette forme (`on_policy_runner.py`).
_CHECKPOINT_GLOB = "model_*.pt"
_VIDEO_GLOB = "videos/*.mp4"


def _tracking_cfg(config: dict) -> dict:
    """Renvoie le nœud ``tracking`` de la config, jamais ``None``."""

    return config.get("tracking") or {}


def _default_task_name(config: dict, log_dir: str) -> str:
    """Nom lisible dans l'UI : le dossier de run porte déjà date, seed et lr."""

    return os.path.basename(os.path.normpath(log_dir)) or "run"


def _default_tags(config: dict) -> list[str]:
    """Étiquettes déduites de la config, pour filtrer les runs dans l'UI.

    On expose ce qui distingue réellement deux expériences de ce dépôt :
    l'espace d'action, et l'état de la physique de rotation.
    """

    spin = config.get("spin") or {}
    tags = [f"action:{config.get('action_type', '?')}"]
    if spin.get("enabled", False):
        tags.append(f"spin:{spin.get('command_mode', '?')}")
        if not spin.get("planner_enabled", True):
            tags.append("ablation:no-planner")
        if not spin.get("magnus_enabled", True):
            tags.append("ablation:no-magnus")
        if not spin.get("analytic_contact_override", True):
            tags.append("ablation:native-contact")
    else:
        tags.append("spin:off")
    return tags


class CheckpointUploader:
    """Pousse vers ClearML les checkpoints et vidéos au fil de leur écriture.

    Pourquoi un guetteur plutôt qu'un appel dans le runner : ``OnPolicyRunner``
    sauvegarde en interne, sans point d'extension, et le patcher pour le suivi
    ferait payer à tout le dépôt le prix d'une fonctionnalité optionnelle. Un
    thread qui surveille ``log_dir`` reste découplé et couvre aussi les fichiers
    écrits par un chemin qu'on n'aurait pas prévu.

    L'intervalle par défaut est large : un checkpoint arrive toutes les
    ``save_interval`` itérations, soit des minutes, et interroger le disque plus
    souvent ne gagnerait rien.
    """

    def __init__(self, task, log_dir: str, *, interval_s: float = 60.0, upload_videos: bool = True):
        self.task = task
        self.log_dir = log_dir
        self.interval_s = interval_s
        self.upload_videos = upload_videos
        self._seen: set[str] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _pending(self) -> list[str]:
        import glob

        patterns = [_CHECKPOINT_GLOB] + ([_VIDEO_GLOB] if self.upload_videos else [])
        found: list[str] = []
        for pattern in patterns:
            found.extend(glob.glob(os.path.join(self.log_dir, pattern)))
        return sorted(p for p in found if p not in self._seen)

    def _upload(self, path: str) -> None:
        # Marqué vu AVANT l'envoi : un fichier qui fait échouer l'upload ne doit
        # pas être retenté à chaque tour de boucle jusqu'à la fin du run.
        self._seen.add(path)
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            if path.endswith(".pt"):
                # Registre *Models* plutôt qu'artefact : le modèle apparaît dans
                # la section MODELS du projet et reste référençable depuis une
                # autre task (reprise, évaluation).
                from clearml import OutputModel

                model = OutputModel(task=self.task, name=name, framework="PyTorch")
                model.update_weights(weights_filename=path, auto_delete_file=False)
            else:
                self.task.upload_artifact(name=name, artifact_object=path, wait_on_upload=False)
            logger.info("ClearML: %s envoyé.", os.path.basename(path))
        except Exception as exc:  # noqa: BLE001 — un envoi raté ne tue pas un run
            logger.warning("ClearML: échec de l'envoi de %s (%s).", path, exc)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            for path in self._pending():
                self._upload(path)

    def start(self) -> None:
        if self.task is None or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="clearml-uploader", daemon=True)
        self._thread.start()

    def flush(self) -> None:
        """Envoie ce qui reste. À appeler avant de rendre la machine."""

        if self.task is None:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        for path in self._pending():
            self._upload(path)


def init_tracking(config: dict, log_dir: str, *, rank: int = 0, task_type: str = "training"):
    """Crée la Task ClearML du run, ou renvoie ``None``.

    À appeler **avant** la construction de ``OnPolicyRunner`` : c'est
    ``Task.init`` qui instrumente TensorBoard, et un ``SummaryWriter`` déjà
    ouvert ne serait pas capté.

    Args:
        config: configuration complète du run (le nœud ``tracking`` la pilote).
        log_dir: dossier de sortie, utilisé pour nommer la task.
        rank: rang du processus. En multi-GPU seul le rang 0 crée la Task —
            les autres remonteraient des courbes concurrentes sur le même nom.
        task_type: ``training`` ou ``testing``, affiché dans l'UI.
    """

    cfg = _tracking_cfg(config)
    if not cfg.get("enabled", False):
        return None
    if rank != 0:
        return None

    try:
        from clearml import Task
    except ImportError:
        logger.warning("clearml n'est pas installé : suivi désactivé.")
        return None

    try:
        if cfg.get("offline", False):
            # Hors-ligne : tout est écrit localement et réimportable plus tard
            # avec Task.import_offline_session. Utile sans réseau sortant.
            Task.set_offline(offline_mode=True)

        task = Task.init(
            project_name=cfg.get("project", "ping_rl"),
            task_name=cfg.get("task_name") or _default_task_name(config, log_dir),
            task_type=task_type,
            tags=list(cfg.get("tags") or []) + _default_tags(config),
            # pytorch=False : on décide nous-mêmes quels poids partent (voir
            # CheckpointUploader). Sans cela ClearML capte chaque torch.save,
            # ce qui double le volume envoyé.
            auto_connect_frameworks={"pytorch": False},
            # output_uri=True : sans destination d'upload, OutputModel
            # enregistre le CHEMIN LOCAL des poids au lieu de les téléverser —
            # le modèle devient inaccessible dès l'instance détruite.
            output_uri=True,
            reuse_last_task_id=False,
        )
        task.connect_configuration(config, name="run")
        logger.info("ClearML: task '%s' (projet '%s').", task.name, task.get_project_name())
        return task
    except Exception as exc:  # noqa: BLE001 — le suivi ne bloque jamais un run
        logger.warning(
            "ClearML indisponible (%s). L'entraînement continue sans suivi. "
            "Configurez les CLEARML_API_* ou passez tracking.enabled=false.",
            exc,
        )
        return None


def report_run_cost(task, *, started_at: float) -> None:
    """Remonte la durée du run et son coût, si le prix horaire est connu.

    ``VAST_DPH`` est injecté par ``scripts/vast.sh`` au lancement : l'API vast
    ne permet pas de retrouver une offre par son identifiant après coup, et les
    variables d'environnement sont figées à la création de l'instance. Sans
    cette variable on remonte quand même la durée, qui est toujours utile.
    """

    if task is None:
        return
    hours = (time.time() - started_at) / 3600.0
    try:
        task.get_logger().report_single_value("train_hours", round(hours, 4))
        dph = os.environ.get("VAST_DPH")
        if dph:
            task.get_logger().report_single_value("train_cost_usd", round(hours * float(dph), 4))
    except Exception as exc:  # noqa: BLE001
        logger.warning("ClearML: impossible de remonter le coût (%s).", exc)


def close_tracking(task, uploader: CheckpointUploader | None = None) -> None:
    """Vide les files d'envoi et clôt la task.

    Indispensable avant ``--auto-destroy`` : ClearML envoie en arrière-plan, et
    une instance détruite trente secondes après la fin du script emporterait
    les derniers fichiers avec elle.
    """

    if uploader is not None:
        uploader.flush()
    if task is None:
        return
    try:
        task.flush(wait_for_uploads=True)
        task.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("ClearML: clôture imparfaite (%s).", exc)
