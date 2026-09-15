#!/usr/bin/env python
"""Profile un rollout sur GPU, et dit où part réellement le temps.

Pourquoi ce script existe : sur les runs de septembre 2026, la collecte
occupait 99 % du temps pour 15-27 % d'utilisation GPU, et le temps par
itération passait de ~3 s à ~40 s dans les dix premières itérations. Deux
diagnostics successifs, tirés de mesures faites sur CPU, se sont révélés faux —
sur CPU la physique domine et un `.item()` ne coûte rien, donc la machine de
développement ne peut pas répondre à la question. Il faut mesurer sur la cible.

Ce que le script établit, et que rien d'autre n'établit :

1. **Le contraste en phase / désynchronisé.** Juste après un reset complet,
   tous les environnements sont en phase et presque aucun ne se termine : les
   branches coûteuses de `step()` ne se déclenchent pas. Une fois désynchronisés,
   une trentaine d'environnements se terminent à CHAQUE pas, et tout se
   déclenche en permanence. C'est très exactement la rampe observée, et le
   script mesure les deux régimes séparément.

2. **Le découpage par section, en temps GPU réel** (`torch.cuda.synchronize`
   autour de chaque section, sans quoi on mesure des lancements asynchrones et
   non du travail).

3. **Le nombre de lancements de noyaux par pas.** C'est le discriminant : des
   milliers de tout petits noyaux signent un coût de latence, que seul un
   regroupement corrige ; quelques gros noyaux signent un coût de calcul, qui
   demande une autre réponse.

Le rapport part dans ClearML : l'instance s'autodétruit et les logs de vast
sont tronqués.

    python scripts/profile_gpu.py --num-envs 1024
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--action-type", default="joint_pd", choices=["joint_pd", "muscle_pd"])
    parser.add_argument("--steps", type=int, default=20, help="pas mesurés par régime")
    parser.add_argument("--warmup", type=int, default=150, help="pas pour désynchroniser")
    parser.add_argument("--trace-steps", type=int, default=3, help="pas capturés dans la trace")
    parser.add_argument(
        "--compare-compiled",
        action="store_true",
        help="mesure le régime établi SANS puis AVEC compilation, dans le même processus",
    )
    parser.add_argument("--out", default="/workspace/profile")
    args = parser.parse_args()

    os.environ.setdefault("GIT_CONFIG_GLOBAL", os.devnull)
    import torch
    import yaml
    from tabletennis_env import TableTennisWarpEnv, tabletennis_p2_cfg
    from train_multigpu import apply_environment_config

    if not torch.cuda.is_available():
        print("✗ Aucun GPU : ce script n'a de sens que sur la cible.", file=sys.stderr)
        return 2
    device = "cuda:0"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load((REPO_ROOT / "default_config.yaml").read_text(encoding="utf-8"))
    config.setdefault("tracking", {})
    config["tracking"]["task_name"] = f"profil-{args.action_type}-{args.num_envs}env"
    task = None
    try:
        from clearml_tracking import init_tracking

        task = init_tracking(config, str(out_dir), task_type="testing")
    except Exception as exc:  # pragma: no cover - dépend du réseau
        print(f"(suivi indisponible : {exc})")

    env_cfg = apply_environment_config(tabletennis_p2_cfg(), config)
    env_cfg.num_envs = args.num_envs
    env_cfg.action_type = args.action_type
    # En mode comparaison, on part SANS compilation : c'est la référence, et
    # l'activer plus tard dans le même processus donne un A/B sur la même
    # machine. Comparer deux locations différentes n'aurait rien prouvé — les
    # hôtes varient trop.
    if args.compare_compiled:
        env_cfg.compile_flight = False
    env = TableTennisWarpEnv(env_cfg, device=device)
    env.reset()
    env.curr_iter, env.total_iter = 40, 100  # curriculum à fond : le cas coûteux

    # --- instrumentation ---------------------------------------------------
    # On enveloppe les méthodes plutôt que d'instrumenter le code de
    # production : le profileur ne doit rien laisser derrière lui.
    timings: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    extra: dict[str, float] = defaultdict(float)

    def wrap(owner, name, label):
        original = getattr(owner, name)

        def timed(*a, **k):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = original(*a, **k)
            torch.cuda.synchronize()
            timings[label] += time.perf_counter() - start
            counts[label] += 1
            return result

        setattr(owner, name, timed)
        return original

    wrap(env.sim, "step", "physique (sim.step)")
    wrap(env.sim, "forward", "sim.forward")
    wrap(env, "_reset_idx", "reset")
    wrap(env, "get_high_command", "planificateur")
    wrap(env, "_cal_reward", "récompense")
    wrap(env, "_update_current_obs", "observations")

    # Combien d'environnements se terminent par pas : c'est ce qui bascule
    # entre les deux régimes.
    reset_sizes: list[int] = []
    inner_reset = env._reset_idx

    def counting_reset(env_ids):
        reset_sizes.append(int(env_ids.numel()))
        return inner_reset(env_ids)

    env._reset_idx = counting_reset

    def rollout(n):
        for _ in range(n):
            env.step(torch.randn(env.num_envs, env.num_actions, device=device) * 0.2)

    def measure(label):
        timings.clear()
        counts.clear()
        reset_sizes.clear()
        torch.cuda.synchronize()
        start = time.perf_counter()
        rollout(args.steps)
        torch.cuda.synchronize()
        wall = time.perf_counter() - start
        accounted = sum(timings.values())
        rows = sorted(timings.items(), key=lambda kv: -kv[1])
        report = [
            f"=== {label} : {args.steps} pas, {args.num_envs} environnements, {args.action_type} ===",
            f"  mur : {wall:.3f} s  ({1000 * wall / args.steps:.1f} ms/pas)",
            f"  environnements resetés par pas : {sum(reset_sizes) / args.steps:.1f} "
            f"(appels : {len(reset_sizes)} pour {args.steps} pas)",
            "",
            f"  {'section':28s} {'temps (s)':>10} {'part':>7} {'appels':>8} {'ms/appel':>10}",
        ]
        for name, seconds in rows:
            report.append(
                f"  {name:28s} {seconds:10.3f} {100 * seconds / wall:6.1f}% "
                f"{counts[name]:8d} {1000 * seconds / max(counts[name], 1):10.2f}"
            )
        report.append(
            f"  {'(non attribué)':28s} {wall - accounted:10.3f} {100 * (wall - accounted) / wall:6.1f}%"
        )
        extra[f"{label}/wall_ms_par_pas"] = 1000 * wall / args.steps
        for name, seconds in rows:
            extra[f"{label}/{name}"] = 1000 * seconds / args.steps
        return "\n".join(report), wall

    lines: list[str] = []
    print("Régime 1 : environnements en phase (juste après un reset complet)", flush=True)
    text, wall_phase = measure("en phase")
    lines.append(text)
    print(text, flush=True)

    print(f"\nDésynchronisation ({args.warmup} pas)…", flush=True)
    rollout(args.warmup)

    print("\nRégime 2 : régime établi", flush=True)
    text, wall_steady = measure("désynchronisé")
    lines.append(text)
    print(text, flush=True)

    if args.compare_compiled:
        import ball_physics

        print("\nActivation de la compilation du pas de vol…", flush=True)
        if ball_physics.enable_compiled_flight(True):
            # Les premiers pas paient la compilation. Mesuré en local :
            # ~32 graphes, les vingt premiers pas à 5-10 s, stabilisé ensuite.
            # On chauffe largement au-delà avant de mesurer quoi que ce soit —
            # mesurer pendant la compilation donnerait un résultat absurde et
            # défavorable.
            rollout(60)
            text, wall_compiled = measure("désynchronisé + compilé")
            lines.append(text)
            print(text, flush=True)
            gain = wall_steady / wall_compiled if wall_compiled else float("nan")
            verdict_compile = [
                "",
                f"=== compilation : x{gain:.2f} sur le régime établi ===",
                f"  {1000 * wall_steady / args.steps:.0f} ms/pas -> "
                f"{1000 * wall_compiled / args.steps:.0f} ms/pas",
            ]
            lines += verdict_compile
            print("\n".join(verdict_compile), flush=True)
            extra["gain_compilation"] = gain
        else:
            print("(compilation indisponible : comparaison impossible)", flush=True)

    ratio = wall_steady / wall_phase if wall_phase else float("nan")
    verdict = [
        "",
        f"=== rampe mesurée : x{ratio:.1f} entre les deux régimes ===",
        "  (les runs montraient ~3 s par itération au départ contre ~40 s au palier)",
    ]
    lines += verdict
    print("\n".join(verdict), flush=True)

    # --- lancements de noyaux ---------------------------------------------
    # Le discriminant : beaucoup de très petits noyaux = coût de latence,
    # que seul un regroupement corrige ; peu de gros noyaux = coût de calcul.
    from torch.profiler import ProfilerActivity, profile

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        rollout(args.trace_steps)
    events = prof.key_averages()
    cuda_events = [e for e in events if getattr(e, "device_time_total", 0) > 0]
    launches = sum(e.count for e in cuda_events)
    cuda_total_ms = sum(e.device_time_total for e in cuda_events) / 1000.0
    kernels = [
        f"  {e.key[:58]:58s} {e.count:7d} {e.device_time_total / 1000.0:9.1f} ms"
        for e in sorted(cuda_events, key=lambda e: -e.device_time_total)[:15]
    ]
    block = [
        "",
        f"=== noyaux CUDA sur {args.trace_steps} pas ===",
        f"  lancements                 : {launches} ({launches / args.trace_steps:.0f} par pas)",
        f"  temps GPU cumulé           : {cuda_total_ms:.1f} ms "
        f"({cuda_total_ms / args.trace_steps:.1f} ms/pas)",
        f"  durée moyenne d'un noyau   : {1000 * cuda_total_ms / max(launches, 1):.1f} µs",
        "",
        f"  {'noyau':58s} {'appels':>7} {'temps':>12}",
        *kernels,
    ]
    lines += block
    print("\n".join(block), flush=True)

    memory = [
        "",
        "=== mémoire GPU ===",
        f"  allouée (pic)  : {torch.cuda.max_memory_allocated() / 2**30:.2f} Go",
        f"  réservée (pic) : {torch.cuda.max_memory_reserved() / 2**30:.2f} Go",
        f"  totale carte   : {torch.cuda.get_device_properties(0).total_memory / 2**30:.2f} Go",
    ]
    lines += memory
    print("\n".join(memory), flush=True)

    report_path = out_dir / f"profil-{args.action_type}-{args.num_envs}env.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    trace_path = out_dir / f"trace-{args.action_type}-{args.num_envs}env.json"
    try:
        prof.export_chrome_trace(str(trace_path))
    except Exception as exc:  # pragma: no cover
        print(f"(trace non exportée : {exc})")

    if task is not None:
        try:
            task.upload_artifact("profil", artifact_object=str(report_path))
            if trace_path.exists():
                task.upload_artifact("trace", artifact_object=str(trace_path))
            logger = task.get_logger()
            for name, value in extra.items():
                logger.report_single_value(name.replace(" ", "_"), round(value, 3))
            logger.report_single_value("rampe", round(ratio, 2))
            logger.report_single_value("lancements_par_pas", round(launches / args.trace_steps, 1))
            logger.report_single_value(
                "memoire_pic_go", round(torch.cuda.max_memory_allocated() / 2**30, 2)
            )
            task.flush(wait_for_uploads=True)
            task.close()
        except Exception as exc:  # pragma: no cover
            print(f"(envoi ClearML partiel : {exc})")

    print(f"\n✓ rapport : {report_path}")
    print(json.dumps({k: round(v, 2) for k, v in extra.items()}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
