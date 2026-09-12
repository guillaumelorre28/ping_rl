# Expériences sur vast.ai, résultats dans ClearML

Trois pièces : une image Docker GPU, un script de pilotage vast.ai, et un
suivi ClearML qui survit à la destruction de la machine.

## Démarrage rapide

```bash
cp .env.example .env          # renseigner CLEARML_API_ACCESS_KEY / SECRET_KEY
uv tool install vastai && vastai set api-key <clé>

docker build --platform linux/amd64 -t glo28/ping-rl:latest .
docker push glo28/ping-rl:latest

./scripts/vast.sh search                              # offres, triées par coût total
VAST_DPH=0.35 ./scripts/vast.sh launch <offer_id> --seed 1
./scripts/vast.sh logs <instance_id>
./scripts/vast.sh destroy <instance_id>               # la facturation court !
```

`--platform linux/amd64` n'est pas optionnel depuis un Mac : une image arm64 ne
démarre pas sur l'hôte, et l'erreur n'apparaît qu'une fois la machine louée.

## Comment le suivi s'accroche

`OnPolicyRunner` écrit ses scalaires dans un `SummaryWriter` TensorBoard.
ClearML instrumente TensorBoard au moment du `Task.init` : il suffit donc de
créer la Task **avant** que le runner ne construise son writer, et toutes les
courbes remontent d'elles-mêmes. C'est ce que fait `train_multigpu.run_train`,
et c'est pourquoi `on_policy_runner.py` n'a pas été modifié.

Conséquence à retenir : **garder `logger: tensorboard`** dans la config.
Choisir `wandb` ou `neptune` court-circuite la capture, et ClearML ne recevrait
plus que la configuration et les artefacts.

En multi-GPU, seul le rang 0 crée la Task : plusieurs rangs produiraient des
runs concurrents écrivant les mêmes courbes sous le même nom.

## Ce qui part vers ClearML, et quand

| Quoi | Quand | Où dans l'UI |
|---|---|---|
| Config complète du run | au démarrage | CONFIGURATION > run |
| Courbes (récompenses, pertes, épisodes) | en continu | SCALARS |
| `model_<it>.pt` | dès l'écriture, toutes les `upload_interval_s` | MODELS |
| Vidéos d'évaluation | idem | ARTIFACTS |
| `train_hours`, `train_cost_usd` | à la fin | SCALARS (valeurs uniques) |

Les checkpoints partent **pendant** le run, pas seulement à la fin : un
entraînement PPO dure des heures, et une instance interrompue ou détruite ne
laisse pas de seconde chance. Un guetteur (`CheckpointUploader`) surveille le
dossier de run sur un thread démon plutôt que d'instrumenter le runner — la
fonctionnalité reste optionnelle et découplée.

Le coût suppose `VAST_DPH`, le prix horaire de l'offre. L'API vast ne permet
pas de retrouver une offre par son identifiant après coup, et les variables
d'environnement sont figées à la création de l'instance : il faut donc le
passer au lancement. Sans lui, seule la durée est remontée.

### Le suivi ne fait jamais échouer un run

ClearML absent, mal configuré ou injoignable produit un avertissement, et
l'entraînement continue. Un envoi qui échoue n'est pas retenté. C'est délibéré :
perdre une courbe est ennuyeux, perdre six heures de GPU ne l'est pas.

Pour travailler sans serveur : `tracking.offline: true` écrit tout localement,
réimportable ensuite avec `Task.import_offline_session`. Pour couper
complètement : `tracking.enabled: false`.

## Files de runs

Lancer une campagne d'ablations sur une seule instance :

```bash
python scripts/run_jobs.py jobs/example.yaml --dry-run    # valider
VAST_DPH=0.35 ./scripts/vast.sh queue <offer_id> jobs/example.yaml --auto-destroy
```

`train_multigpu.py` n'expose que cinq arguments, alors qu'une ablation doit
pouvoir toucher n'importe quelle clé. Chaque job écrit donc **sa propre config
dérivée** — config de base plus surcharges pointées — passée via `--config` :

```yaml
jobs:
  - name: ablation-sans-magnus
    args: {seed: 1}
    overrides:
      spin.magnus_enabled: false
```

Une surcharge dont la clé n'existe pas dans la config de base est **refusée à
la validation**. Une ablation qui ne s'applique pas silencieusement produirait
un résultat qu'on croirait informatif et qui ne le serait pas.

Chaque job tourne dans un sous-processus : un run qui plante n'emporte pas la
file. `--keep-going` poursuit malgré un échec (c'est ce qu'utilise `vast.sh
queue`). Le nom du job devient le nom de la task ClearML.

Le fichier est compressé puis encodé dans la commande de démarrage : changer la
file ne demande aucun rebuild d'image. L'API vast plafonne cette commande à
16 384 caractères, et les commentaires comptent — un fichier très commenté peut
devoir être scindé. `vast.sh` vérifie la longueur avant de louer.

## `--auto-destroy`

L'instance se détruit seule à la fin du travail, ce qui arrête la facturation.
Le script **refuse** de l'activer sans `CLEARML_API_*` : sans suivi, les poids
mourraient avec la machine. Trente secondes séparent la fin du run de la
destruction, pour laisser ClearML finir ses envois — et `close_tracking` est
appelé dans un `finally`, donc même un run interrompu dépose ce qu'il a.

Vérifiez tout de même avec `./scripts/vast.sh list` : l'autodestruction dépend
d'un appel API qui peut échouer.

## Choisir une offre

Les filtres par défaut de `vast.sh search` encodent chacun une panne concrète,
documentée dans le script. Les deux qui coûtent le plus cher quand on les
assouplit :

- **`cuda_vers`** doit suivre la variante CUDA de torch dans l'image. Sur un
  hôte au pilote trop ancien, `torch.cuda.is_available()` est faux, MuJoCo-Warp
  retombe sur CPU, et l'entraînement rampe des heures au tarif GPU. À
  revérifier après chaque rebuild :
  ```bash
  docker run --rm glo28/ping-rl:latest python -c "import torch; print(torch.version.cuda)"
  ```
- **`gpu_frac=1`** prend la machine entière. Une fraction de machine multi-GPU
  peut voir ses ressources prises entre la réservation et le démarrage :
  l'instance part en file d'attente indéfinie, disque facturé.

Le tri se fait sur le coût **total** (calcul + stockage) : le stockage varie
d'un facteur quatre entre hôtes et renverse régulièrement le classement.

## Sans ClearML

`./scripts/vast.sh copy <instance_id>` rapatrie `/app/logs` (checkpoints,
vidéos, événements TensorBoard) avant destruction. À faire **avant**
`destroy`, et sans `--auto-destroy`.

## L'image

Construite pour vast.ai : `tini` en PID 1 (sinon les processus `torchrunx`
survivent à un Ctrl-C et l'instance continue de facturer), `openssh-server`
pour `vast.sh ssh`, `libegl-dev` pour le rendu hors écran des vidéos
d'évaluation.

Les dépendances sont installées en deux passes pour produire deux couches
plutôt qu'une très grosse : Docker télécharge plusieurs couches en parallèle,
et un pull interrompu ne reprend pas tout depuis zéro. Le modèle et les meshes
(~60 Mo, immuables) sont dans une couche séparée du code, qui change à chaque
itération.

`.dockerignore` exclut le `.venv` — près d'un Go qui partait jusqu'ici dans le
contexte de build — ainsi que `.git`, les sorties d'entraînement et le `.env`.
Une image publiée ne doit pas embarquer les clés ClearML.
