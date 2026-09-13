# Image d'entraînement GPU (vast.ai, RunPod, ou toute machine NVIDIA).
#
# Build depuis un Mac — cibler l'architecture des GPU cloud, sinon l'image
# produite est arm64 et ne démarre pas sur l'hôte :
#   docker build --platform linux/amd64 -t glo28/ping-rl:latest .
#   docker push glo28/ping-rl:latest
#
# Run local avec GPU :
#   docker run --rm -it --gpus all \
#     -v $PWD/logs:/app/logs \
#     -e CLEARML_API_ACCESS_KEY -e CLEARML_API_SECRET_KEY \
#     glo28/ping-rl:latest \
#     python train_multigpu.py --gpu-ids 0
#
# Vérifier que la variante CUDA de torch correspond au pilote des hôtes visés
# (le filtre `cuda_vers` de scripts/vast.sh doit suivre) :
#   docker run --rm glo28/ping-rl:latest python -c "import torch; print(torch.version.cuda)"

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PYTHON=3.13 \
    MUJOCO_GL=egl

# libegl-dev : MuJoCo rend hors écran via EGL (MUJOCO_GL=egl) — sans lui, la
#   capture vidéo d'évaluation échoue au premier eval_interval, après des
#   heures d'entraînement déjà payées.
# openssh-server : requis par le mode SSH de vast.ai (`vast.sh ssh`).
# tini : PID 1 correct, sinon les processus torchrun deviennent orphelins et
#   l'instance continue de facturer après un Ctrl-C.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates libegl-dev openssh-server tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 1) Dépendances seules : couche mise en cache tant que uv.lock ne bouge pas.
# Installées en deux passes pour produire deux layers de taille comparable
# plutôt qu'un seul très gros : Docker télécharge plusieurs layers en
# parallèle, et un pull interrompu ne reprend pas tout depuis zéro — ce qui
# compte sur des hôtes cloud dont le débit descendant est inégal.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install ${UV_PYTHON} && \
    uv sync --locked --no-install-project --no-editable --no-dev \
        --no-install-package torch
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-editable --no-dev

# 2) Modèle et ressources. Séparés du code : les meshes MyoSuite pèsent ~60 Mo
# et ne changent jamais, alors que les sources changent à chaque itération.
# Le XML référence ces deux dossiers en chemins relatifs (meshdir="./").
COPY myo_sim ./myo_sim
COPY assets ./assets
COPY tabletennis.xml ./

# 3) Le code, couche légère invalidée à chaque modification.
COPY src ./src
COPY scripts ./scripts
COPY jobs ./jobs
COPY tests ./tests
COPY default_config.yaml ./
COPY ball_physics.py planner.py tabletennis_env.py on_policy_runner.py \
     muscle_utils.py clearml_tracking.py train_multigpu.py ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --no-dev

# pytest dans l'image de production : trois défauts (dépendance non déclarée,
# drapeau MuJoCo renommé, runner incompatible avec rsl_rl) ont été découverts
# sur une machine louée parce que rien n'exerçait l'artefact avant de payer.
# `make docker-verify` fait tourner la suite et scripts/preflight.py ICI, sur
# l'image exacte qui sera lancée. Quelques mégaoctets contre une location.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install pytest

# Le venv en tête de PATH : `python`, `torchrun` et `pytest` utilisables tels
# quels dans --onstart-cmd, sans préfixe `uv run`.
ENV PATH="/app/.venv/bin:${PATH}"

# Warp compile ses noyaux au premier pas et les met en cache ici. Sur un
# volume, le second run d'une même machine démarre sans recompiler.
ENV WARP_CACHE_PATH=/workspace/warp_cache
RUN mkdir -p /workspace

ENTRYPOINT ["tini", "--"]
CMD ["python", "tests/smoke_test.py"]
