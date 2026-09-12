#!/usr/bin/env bash
# Pilotage d'entraînements table tennis sur vast.ai avec l'image du projet.
#
# Prérequis (une seule fois) :
#   uv tool install vastai
#   vastai set api-key <clé>          # https://cloud.vast.ai -> Account -> API Key
#   cp .env.example .env              # CLEARML_API_*, VAST_IMAGE si autre image
#
# Usage :
#   ./scripts/vast.sh search [critères vastai]      # offres GPU triées par coût total
#   ./scripts/vast.sh launch <offer_id> [--auto-destroy] [args de train_multigpu.py]
#   ./scripts/vast.sh queue  <offer_id> <jobs.yaml> [--auto-destroy]
#       exécute une liste de runs en séquence (voir jobs/example.yaml) ; le
#       fichier local est embarqué dans la commande de démarrage, donc changer
#       la file ne demande AUCUN rebuild d'image
#   ./scripts/vast.sh list                          # instances en cours
#   ./scripts/vast.sh logs <instance_id>
#   ./scripts/vast.sh ssh <instance_id>
#   ./scripts/vast.sh copy <instance_id> [dest]     # rapatrie /app/logs
#   ./scripts/vast.sh destroy <instance_id>
#
# --auto-destroy : l'instance se détruit TOUTE SEULE à la fin du travail, ce qui
#   arrête la facturation. Exige CLEARML_API_* : sans suivi, les poids meurent
#   avec la machine.
#
# Exemples :
#   ./scripts/vast.sh search
#   ./scripts/vast.sh launch 1234567 --gpu-ids all --seed 1
#   VAST_DPH=0.35 ./scripts/vast.sh queue 1234567 jobs/example.yaml --auto-destroy
#
# Les variables CLEARML_API_* (du shell, sinon du .env) sont transmises à
# l'instance. Les checkpoints et vidéos partent vers ClearML PENDANT le run
# (clearml_tracking.CheckpointUploader) : un run interrompu reste récupérable.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Credentials et réglages du .env à la racine (non versionné, voir
# .env.example). Les variables déjà exportées dans le shell gardent la
# priorité. Chargé AVANT la lecture de VAST_IMAGE/VAST_DISK_GB.
if [[ -f "$REPO_ROOT/.env" ]]; then
  while IFS='=' read -r key value; do
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    [[ -n "${!key:-}" ]] || export "$key=$value"
  done < "$REPO_ROOT/.env"
fi

IMAGE="${VAST_IMAGE:-docker.io/glo28/ping-rl:latest}"
DISK_GB="${VAST_DISK_GB:-48}"

usage() { awk 'NR>1 && !/^#/{exit} NR>1{sub(/^# ?/,""); print}' "$0"; exit 1; }
[[ $# -ge 1 ]] || usage
command -v vastai >/dev/null 2>&1 || {
  echo "vastai introuvable. Installez-le : uv tool install vastai && vastai set api-key <clé>" >&2
  exit 1
}

# Clé API vast : variable d'environnement, sinon celle enregistrée par le CLI.
vast_api_key() {
  if [[ -n "${VAST_API_KEY:-}" ]]; then echo "$VAST_API_KEY"; return; fi
  local f
  for f in "$HOME/.config/vastai/vast_api_key" "$HOME/.vast_api_key"; do
    if [[ -f "$f" ]]; then tr -d '[:space:]' < "$f"; return; fi
  done
}

AUTODESTROY=0
ENV_ARGS=""

# Consomme les options --xxx connues ; le reste est rendu dans REMAINING pour
# être passé tel quel à train_multigpu.py (qui a ses propres --gpu-ids, --seed…).
parse_flags() {
  REMAINING=()
  while [[ $# -ge 1 ]]; do
    case "$1" in
      --auto-destroy) AUTODESTROY=1 ;;
      *) REMAINING+=("$1") ;;
    esac
    shift
  done
}

build_env_args() {
  ENV_ARGS="-e MUJOCO_GL=egl"
  local var
  # VAST_DPH : prix horaire de la machine. L'API vast ne permet pas de
  # retrouver une offre par son id après coup, et les variables d'environnement
  # sont figées à la création de l'instance : le prix ne peut donc pas être
  # découvert plus tard. C'est à l'appelant de le passer — il l'a sous les yeux,
  # colonne TOTAL de `search`.
  for var in CLEARML_API_ACCESS_KEY CLEARML_API_SECRET_KEY \
             CLEARML_API_HOST CLEARML_WEB_HOST CLEARML_FILES_HOST VAST_DPH; do
    if [[ -n "${!var:-}" ]]; then
      ENV_ARGS+=" -e ${var}=${!var}"
    fi
  done
  if [[ -z "${VAST_DPH:-}" ]]; then
    echo "ℹ VAST_DPH non défini : la métrique train_cost_usd ne sera pas calculée." >&2
    echo "  Relancez avec le prix horaire de l'offre (colonne TOTAL de '$0 search') :" >&2
    echo "    VAST_DPH=0.35 $0 ..." >&2
  fi
  if [[ $AUTODESTROY -eq 1 ]]; then
    if [[ -z "${CLEARML_API_ACCESS_KEY:-}" ]]; then
      echo "✗ --auto-destroy sans CLEARML_API_* : les poids seraient détruits avec l'instance." >&2
      echo "  Exportez CLEARML_API_ACCESS_KEY / CLEARML_API_SECRET_KEY, ou retirez --auto-destroy." >&2
      exit 1
    fi
    local key
    key=$(vast_api_key)
    if [[ -z "$key" ]]; then
      echo "✗ --auto-destroy : clé API vast introuvable (export VAST_API_KEY=... ou vastai set api-key)." >&2
      exit 1
    fi
    ENV_ARGS+=" -e VAST_API_KEY=${key}"
  elif [[ "$ENV_ARGS" != *CLEARML_API_ACCESS_KEY* ]]; then
    echo "⚠ CLEARML_API_* non définies : pas de suivi ni d'upload des poids." >&2
    echo "  Sans ClearML, rapatriez les logs AVANT destroy : $0 copy <instance_id>" >&2
  fi
}

# Fin de onstart quand --auto-destroy : l'instance s'autodétruit via l'API
# (l'id est fourni par vast : CONTAINER_ID ou VAST_CONTAINERLABEL="C.<id>").
# Les 30 s de délai laissent ClearML finir ses envois en arrière-plan.
destroy_snippet() {
  printf '%s' "; EXIT=\$?; echo \"[vast.sh] travail terminé (exit \$EXIT), autodestruction dans 30 s...\"; \
sleep 30; IID=\${CONTAINER_ID:-\${VAST_CONTAINERLABEL#C.}}; \
curl -s -X DELETE \"https://console.vast.ai/api/v0/instances/\${IID}/?api_key=\${VAST_API_KEY}\" >/dev/null \
&& echo \"[vast.sh] destruction demandée.\" \
|| echo \"[vast.sh] ÉCHEC de l'autodestruction : détruisez l'instance manuellement !\""
}

# Vérifie que l'instance atteint RÉELLEMENT l'état "running".
#
# Pourquoi ne pas se fier à la réponse de l'API : elle ment dans les deux sens.
# success=false alors que le contrat est créé ; success=true alors que
# l'instance reste à l'arrêt, sans conteneur. `list` affiche alors
# actual_status=loading, trompeur — c'est `intended_status` qui fait autorité.
attendre_demarrage() {  # $1 = instance_id
  local iid=$1 essai etat msg intended actual
  for essai in $(seq 1 20); do
    etat=$(vastai show instances --raw 2>/dev/null | python3 -c "
import json, sys
iid = int('$iid')
for x in json.load(sys.stdin):
    if x.get('id') == iid:
        print(x.get('intended_status') or '?', x.get('actual_status') or '?')
        break
else:
    print('absente ?')
" 2>/dev/null || echo "? ?")
    read -r intended actual <<<"$etat"
    intended=${intended:-?}; actual=${actual:-?}

    if [[ "$intended" == "running" ]]; then
      echo "✓ Instance $iid démarrée (intended=running, actual=$actual)."
      return 0
    fi
    if [[ "$intended" == "stopped" ]]; then
      echo "⚠ Instance $iid à l'ARRÊT (intended=stopped) : démarrage explicite..."
      msg=$(vastai start instance "$iid" 2>&1) || true
      echo "  $msg"
      if printf '%s' "$msg" | grep -qi "unavailable"; then
        echo "" >&2
        echo "✗ La machine hôte n'a plus de GPU libre : votre demande part en" >&2
        echo "  FILE D'ATTENTE, sans délai garanti, et le disque est facturé." >&2
        echo "  Détruisez et prenez une autre offre (préférez une machine" >&2
        echo "  MONO-GPU, sans colocataires pour se disputer les ressources) :" >&2
        echo "    $0 destroy $iid" >&2
        return 1
      fi
    fi
    sleep 15
  done
  echo "" >&2
  echo "✗ Instance $iid toujours pas démarrée après 5 min (intended=$intended)." >&2
  echo "  Vérifiez '$0 list' ; si elle reste bloquée : $0 destroy $iid" >&2
  return 1
}

create_instance() {  # $1 = offer_id, $2 = onstart, $3 = description
  # L'API vast plafonne la commande de démarrage à 16 384 caractères et répond
  # sinon un 400 laconique, APRÈS avoir affiché le récapitulatif — on croit
  # avoir loué, on n'a rien. Mieux vaut échouer ici, en disant quoi faire.
  if [[ ${#2} -gt 16000 ]]; then
    echo "✗ Commande de démarrage trop longue : ${#2} caractères (limite API : 16384)." >&2
    echo "  Allégez le fichier de jobs — les commentaires comptent dans le payload —" >&2
    echo "  ou scindez la file en deux lancements." >&2
    return 1
  fi
  echo "Image     : $IMAGE"
  echo "Disque    : ${DISK_GB} Go"
  echo "Travail   : $3"
  echo "Autodestr.: $([[ $AUTODESTROY -eq 1 ]] && echo oui || echo non)"
  local out iid
  out=$(vastai create instance "$1" \
    --image "$IMAGE" \
    --disk "$DISK_GB" \
    --ssh --direct \
    --env "$ENV_ARGS" \
    --onstart-cmd "$2" \
    --raw)
  echo "$out"

  iid=$(printf '%s' "$out" | python3 -c "
import json, re, sys
raw = sys.stdin.read()
try:
    print(json.loads(raw).get('new_contract') or '')
except Exception:
    m = re.search(r'new_contract\\D*(\\d+)', raw)
    print(m.group(1) if m else '')
" 2>/dev/null || true)

  if [[ -z "$iid" ]]; then
    echo "⚠ Impossible de retrouver l'id de l'instance : vérifiez avec '$0 list'." >&2
  else
    echo "Instance  : $iid"
    attendre_demarrage "$iid" || return 1
  fi
  echo
  echo "Suivi : $0 logs ${iid:-<instance_id>}, ou les courbes ClearML."
  if [[ $AUTODESTROY -eq 1 ]]; then
    echo "Autodestruction active : vérifiez quand même avec '$0 list' à la fin."
  else
    echo "Pensez à '$0 destroy <instance_id>' à la fin — la facturation court !"
  fi
}

cmd=$1
shift
case "$cmd" in
  search)
    # Critères par défaut, chacun issu d'un mode de panne concret —
    # NE PAS les assouplir sans savoir ce qu'on abandonne :
    #
    #   cuda_vers>=${VAST_MIN_CUDA}
    #                    Le torch de l'image est un build CUDA donné. Sur un hôte
    #                    au pilote plus ancien, torch.cuda.is_available() est faux,
    #                    MuJoCo-Warp retombe sur CPU et l'entraînement rampe des
    #                    heures, GPU à 0 %, facturé plein tarif. Revérifier après
    #                    un rebuild :
    #                      docker run --rm $IMAGE python -c \
    #                        "import torch; print(torch.version.cuda)"
    #   gpu_frac=1       La machine ENTIÈRE, sans colocataires. Une fraction de
    #                    machine multi-GPU peut voir ses ressources prises entre
    #                    la réservation et le démarrage : l'instance part en file
    #                    d'attente indéfinie, disque facturé.
    #   compute_cap>=800 Ampere ou plus récent. MuJoCo-Warp génère ses noyaux
    #                    pour l'architecture de l'hôte ; sous Ampere, les gains
    #                    de la simulation massivement parallèle s'effondrent.
    #   disk_space>=...  L'hôte doit pouvoir FOURNIR le disque demandé. Sans ce
    #                    filtre, vast crée le contrat sur une machine qui n'a pas
    #                    la place : l'image ne se décompresse pas ("no space left
    #                    on device"), l'instance reste en loading et facture.
    #   reliability>0.98 Sous ce seuil, les interruptions en cours de run sont
    #                    fréquentes — un entraînement PPO dure des heures.
    #
    # Le tri se fait sur le coût TOTAL (calcul + stockage) : le stockage varie
    # d'un facteur 4 entre hôtes et renverse régulièrement le classement.
    query=${*:-"num_gpus=1 gpu_frac=1 gpu_ram>=16 disk_space>=${VAST_DISK_GB:-48} compute_cap>=800 reliability>0.98 dph<=0.60 cuda_vers>=${VAST_MIN_CUDA:-12.8} rentable=true"}
    vastai search offers "$query" -o 'dph+' --raw | python3 -c "
import json, sys
offres = json.load(sys.stdin)
if not offres:
    print('Aucune offre. Assouplissez un critère — mais lisez d abord les commentaires.')
    raise SystemExit
lignes = sorted(((o['dph_total'] + o.get('storage_cost', 0), o) for o in offres), key=lambda r: r[0])
print(f\"{'ID':>10} {'GPU':<14} {'VRAM':>5} {'calcul':>7} {'disque':>7} {'TOTAL':>7} {'CUDA':>5} {'fiab':>6} {'down':>6}  pays\")
for total, o in lignes[:15]:
    print(f\"{o['id']:>10} {o['gpu_name'][:14]:<14} {o['gpu_ram']/1024:>4.0f}G \"
          f\"{o['dph_total']:>7.3f} {o.get('storage_cost', 0):>7.3f} {total:>7.3f} \"
          f\"{str(o.get('cuda_max_good')):>5} {o.get('reliability2', 0):>6.4f} \"
          f\"{o.get('inet_down', 0):>6.0f}  {str(o.get('geolocation'))[:20]}\")
print()
print(f'{len(offres)} offre(s) ; tri par coût TOTAL (calcul + stockage).')
"
    ;;

  launch)
    [[ $# -ge 1 ]] || { echo "usage: $0 launch <offer_id> [--auto-destroy] [args train_multigpu.py]" >&2; exit 1; }
    offer=$1
    shift
    parse_flags "$@"
    set -- "${REMAINING[@]+"${REMAINING[@]}"}"
    build_env_args

    # train_multigpu.py gère lui-même la topologie GPU (--gpu-ids all passe par
    # torchrunx) : pas de torchrun à composer ici. Par défaut, toutes les cartes.
    args="$*"
    [[ "$args" == *--gpu-ids* ]] || args="--gpu-ids all $args"
    run_cmd="python train_multigpu.py $args"

    onstart="cd /app && set -o pipefail && { $run_cmd ; } 2>&1 | tee /workspace/train.log"
    [[ $AUTODESTROY -eq 1 ]] && onstart+="$(destroy_snippet)"

    echo "Arguments : $args"
    create_instance "$offer" "$onstart" "train"
    ;;

  queue)
    [[ $# -ge 2 ]] || { echo "usage: $0 queue <offer_id> <jobs.yaml> [--auto-destroy]" >&2; exit 1; }
    offer=$1
    jobs_file=$2
    shift 2
    parse_flags "$@"
    [[ -f "$jobs_file" ]] || { echo "Fichier de jobs introuvable : $jobs_file" >&2; exit 1; }

    # Validation locale AVANT de payer une instance (format, clés inconnues).
    (cd "$REPO_ROOT" && .venv/bin/python scripts/run_jobs.py "$jobs_file" --dry-run) || {
      echo "✗ Fichier de jobs invalide (voir erreurs ci-dessus)." >&2; exit 1;
    }
    build_env_args

    # Le fichier local est embarqué dans la commande de démarrage : modifier la
    # file ne demande AUCUN rebuild.
    #
    # GZIPPÉ avant l'encodage : l'API vast refuse toute commande de plus de
    # 16 384 caractères, et un fichier de jobs commenté les dépasse vite.
    # `gzip -n` supprime nom et horodatage — même fichier, même payload.
    jobs_b64=$(gzip -9nc < "$jobs_file" | base64 | tr -d '\n')
    run_cmd="echo $jobs_b64 | base64 -d | gzip -dc > /workspace/jobs.yaml && \
python scripts/run_jobs.py /workspace/jobs.yaml --keep-going"

    onstart="cd /app && set -o pipefail && { $run_cmd ; } 2>&1 | tee /workspace/train.log"
    [[ $AUTODESTROY -eq 1 ]] && onstart+="$(destroy_snippet)"

    create_instance "$offer" "$onstart" "queue $jobs_file"
    ;;

  list)    vastai show instances ;;

  logs)
    [[ $# -ge 1 ]] || { echo "usage: $0 logs <instance_id>" >&2; exit 1; }
    vastai logs "$1"
    ;;

  ssh)
    [[ $# -ge 1 ]] || { echo "usage: $0 ssh <instance_id>" >&2; exit 1; }
    url=$(vastai ssh-url "$1")
    echo "$url"
    hostport=${url#ssh://root@}
    exec ssh -o StrictHostKeyChecking=accept-new -p "${hostport##*:}" "root@${hostport%%:*}"
    ;;

  copy)
    [[ $# -ge 1 ]] || { echo "usage: $0 copy <instance_id> [dest]" >&2; exit 1; }
    dest=${2:-outputs_vast}
    vastai copy "$1:/app/logs" "$dest"
    echo "Logs copiés dans $dest"
    ;;

  destroy)
    [[ $# -ge 1 ]] || { echo "usage: $0 destroy <instance_id>" >&2; exit 1; }
    vastai destroy instance "$1"
    ;;

  *) usage ;;
esac
