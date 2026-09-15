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

## Vérifier avant de louer

```bash
make docker-verify        # suite de tests + assemblage réel, dans l'image
```

Quatre défauts ont été découverts sur une machine louée plutôt qu'ici, chacun
coûtant une location et un aller-retour : une dépendance non déclarée
(`ml_collections`), un drapeau MuJoCo renommé (`mjENBL_MULTICCD`), deux
versions de MuJoCo mutuellement incompatibles, et un `OnPolicyRunner` appelant
une signature que la version épinglée de `rsl_rl` n'avait plus. Les trois
premiers auraient été vus par un simple import ; le quatrième non — la suite de
tests n'instancie jamais le runner.

`scripts/preflight.py` comble ce trou : il refait ce que fait
`train_multigpu.run_train` — environnement, runner, **une itération complète
d'entraînement et l'évaluation avec rendu vidéo** — puis s'arrête, pour les
deux espaces d'action. Sur CPU, sans GPU ni réseau. `docker-push` en dépend,
donc une image qui ne passe pas la barrière ne part pas au registre.

Le préflight est lent sous émulation (`--platform linux/amd64` sur un Mac
Apple Silicon) : compter une vingtaine de minutes, l'essentiel passant dans le
rendu hors écran de l'évaluation. C'est quelques minutes sur l'hôte cible.

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
| `diag/nonfinite_reward_envs`, `diag/nonfinite_obs_envs` | en continu | SCALARS |
| `diag/plan_resamples`, `diag/infeasible_plans` | en continu | SCALARS |

### Les deux courbes `diag/` sont à surveiller

Elles donnent la fraction d'environnements dont l'état a divergé. Elles doivent
rester à zéro ; une qui décolle date l'incident à l'itération près.

Elles existent parce que trois `nan_to_num` muets — deux sur les observations,
un sur la récompense — faisaient passer un environnement corrompu pour un
environnement ordinaire à récompense nulle. Sur le run du 13 septembre 2026,
trois termes sont passés en NaN à l'itération 18 sans jamais en revenir : le
total n'a baissé que de 8 %, donc rien n'a alerté, mais **4,2 points par pas
étaient gagnés sans être journalisés**. ClearML sérialisant un NaN en `0.0`,
les courbes semblaient simplement plates.

Les moyennes du runner écartent désormais les entrées non finies plutôt que de
laisser un seul environnement effacer une série entière, et remontent le
nombre d'épisodes écartés sous `diag/dropped_episodes/`. Quand il ne reste
aucune donnée saine, rien n'est tracé : une absence se lit comme une absence,
un zéro se lit comme une mesure.

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

## Où passe le temps

Sur le run du 14 septembre 2026, la collecte occupait **99,3 %** du temps et
l'apprentissage 0,7 %, pour **27 % d'utilisation GPU**. Le GPU attendait.

La cause était dans `_reset_idx`, qui contenait à lui seul toutes les
synchronisations GPU vers hôte du fichier : deux rejets bouclant sur
`.all().item()`, soit jusqu'à 28 allers-retours sérialisés par reset. Or un
reset a lieu à **chaque pas de contrôle** — avec 2048 environnements et des
épisodes de ~69 pas, une trentaine d'environnements se terminent par pas.

La mesure qui l'a établi : le coût d'un reset était presque indépendant du
nombre d'environnements concernés — 0,46 s pour **un seul**, 0,75 s pour 512.
Multiplier le travail par 512 ne coûtait que 63 % de plus. C'est la signature
d'un coût de latence, pas de calcul.

Deux changements, et `_reset_idx` ne contient plus aucune synchronisation :

- **Vivier de lancers.** Le tirage de la balle entrante ne dépend que du niveau
  de curriculum, jamais de la politique. Il est fait une fois par palier
  (`launch_pool_curriculum_step`), et les resets se servent par indexation.
- **Candidats de commande groupés.** Les relances du planificateur deviennent K
  tirages indépendants évalués en un appel, dont on retient le premier
  faisable — même échantillonnage par rejet, donc même distribution, mais une
  seule passe.

### Ce que le profil a montré

`scripts/profile_gpu.py` sur RTX 3090, 1024 environnements :

| | temps par pas |
|---|---|
| environnements en phase (aucun ne se termine) | 175 ms |
| régime établi | 1866 ms |

Facteur **10,7** — la rampe de 3 s à 40 s par itération vient entièrement de
cette bascule. Découpage en régime établi : **planificateur 88,4 %**, reset
41,9 % (il contient le planificateur), récompense 4,8 %, **physique MuJoCo Warp
2,8 %**. La physique n'était pas le problème.

Cause : **368 412 lancements de noyaux CUDA par pas**, d'une durée moyenne de
**1,3 µs** — très en dessous du coût de lancement. Du calcul numérique
parfaitement correct, mais émis en opérations élémentaires minuscules :
`predict_land` est appelé neuf fois par plan, chacun faisant 4 itérations de
Newton x 3 sous-pas x 4 étages RK, chaque étage réévaluant un modèle
aérodynamique à interpolation par table.

`compile_flight` fusionne ce noyau avec `torch.compile`. Sur un appel complet
au planificateur :

| | opérations | temps |
|---|---|---|
| direct | 140 397 | 57,9 ms |
| pas RK4 compilé | 15 405 | 11,8 ms |
| **propagations compilées** | **8 201** | **6,8 ms** |

Ce sont les propagations entières qui sont compilées, pas seulement le pas
RK4 : la boucle de Newton et les sous-pas s'y retrouvent dans un même graphe.

Les écarts numériques sont ceux du float32. La boucle de Newton les amplifie —
la propagation seule dévie de 1e-7, la commande de raquette de 1e-4 en absolu —
mais rapporté aux échelles (6,9 m/s, 28 rad/s, 338 rad/s) cela fait au plus
**1,2e-5 en relatif**, sept ordres de grandeur sous la randomisation de domaine
(±10 %). Le drapeau de faisabilité, lui, est exigé identique.
`tests/test_compiled_flight.py` verrouille les deux.

`dynamic=True` produit un seul graphe pour toutes les tailles de lot —
indispensable, le nombre d'environnements resetés changeant à chaque pas. Elle
retombe seule sur l'exécution directe si elle échoue.

**Gain mesuré**, A/B dans le même processus sur RTX 3090, 1024 environnements :

```
1144 ms/pas  ->  240 ms/pas     x4,76
planificateur : 559 ms/appel -> 63 ms/appel     x8,8
```

**Et son coût : ~8,5 minutes de compilation au démarrage** (contre ~150 s sur
un poste de développement). Sur 3000 itérations c'est 3,5 % ; sur une
vérification de 100 itérations, cela dominerait le run. D'où le réglage :
mettre `compile_flight: false` pour les runs courts.

Réserve sur la mesure : les deux points n'étaient pas au même niveau de
désynchronisation — 3,5 environnements resetés par pas contre 15,5 après la
chauffe supplémentaire, alors que le régime établi vaut num_envs divisé par la
longueur d'épisode, soit ~15. La variante compilée a donc fait ~11 % d'appels
au planificateur en plus, ce qui sous-estime légèrement le gain. La chauffe de
150 pas du profileur ne suffit pas à atteindre le régime établi.

    ./scripts/vast.sh profile <offer_id> --auto-destroy --compare-compiled

mesure le régime établi sans puis avec compilation **dans le même processus** :
comparer deux locations différentes ne prouverait rien, les hôtes variant trop.

### Le filtrage des commandes est maintenant mesuré

« Plan infaisable » ne veut pas dire NaN : il veut dire que le coup demandé
exigerait une raquette au-delà de ses limites (12 m/s, 30 rad/s). Le rejeter a
donc un sens, mais il biaise la distribution des effets demandés vers ce qui
est atteignable — un biais jusqu'ici invisible, car `target_plan_valid` était
écrit et jamais lu.

`diag/plan_resamples` donne l'indice du candidat retenu (0 = le premier tirage
convenait) et `diag/infeasible_plans` la part des épisodes où aucun des K ne
convenait. Mesuré à curriculum plein : indice moyen **0,134** et **0 %**
d'infaisables avec K=8.

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
