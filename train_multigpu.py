import logging
import os
from datetime import datetime
from typing import Literal

import yaml
from tabletennis_env import TableTennisWarpEnv, tabletennis_p2_cfg

from mjlab.utils.gpu import select_gpus
from mjlab.utils.random import seed_rng


def apply_environment_config(env_cfg, config: dict):
    """Apply user-facing environment overrides from the training YAML."""

    env_cfg.action_type = config.get("action_type", env_cfg.action_type)
    env_cfg.enable_multiccd = config.get("enable_multiccd", env_cfg.enable_multiccd)
    env_cfg.enable_domain_randomization = config.get(
        "enable_domain_randomization", env_cfg.enable_domain_randomization
    )
    spin_cfg = config.get("spin", {})
    if spin_cfg:
        env_cfg.spin_physics_enabled = spin_cfg.get("enabled", env_cfg.spin_physics_enabled)
        env_cfg.analytic_contact_override = spin_cfg.get(
            "analytic_contact_override", env_cfg.analytic_contact_override
        )
        env_cfg.reciprocal_contact_impulse = spin_cfg.get(
            "reciprocal_contact_impulse", env_cfg.reciprocal_contact_impulse
        )
        env_cfg.aerodynamic_model = spin_cfg.get(
            "aerodynamic_model", env_cfg.aerodynamic_model
        )
        env_cfg.drag_enabled = spin_cfg.get("drag_enabled", env_cfg.drag_enabled)
        env_cfg.magnus_enabled = spin_cfg.get("magnus_enabled", env_cfg.magnus_enabled)
        env_cfg.incoming_spin_enabled = spin_cfg.get(
            "incoming_spin_enabled", env_cfg.incoming_spin_enabled
        )
        env_cfg.spin_planner_enabled = spin_cfg.get(
            "planner_enabled", env_cfg.spin_planner_enabled
        )
        env_cfg.spin_reward_enabled = spin_cfg.get(
            "reward_enabled", env_cfg.spin_reward_enabled
        )
        env_cfg.spin_command_mode = spin_cfg.get("command_mode", env_cfg.spin_command_mode)
        env_cfg.spin_target = tuple(spin_cfg.get("target", env_cfg.spin_target))
        if "target_range" in spin_cfg:
            env_cfg.spin_target_range.low = tuple(spin_cfg["target_range"]["low"])
            env_cfg.spin_target_range.high = tuple(spin_cfg["target_range"]["high"])
        if "incoming_range" in spin_cfg:
            env_cfg.incoming_spin_range.low = tuple(spin_cfg["incoming_range"]["low"])
            env_cfg.incoming_spin_range.high = tuple(spin_cfg["incoming_range"]["high"])
        if "reward_sigma" in spin_cfg:
            env_cfg.spin_reward_sigma = tuple(spin_cfg["reward_sigma"])
        if "reward_axis_weights" in spin_cfg:
            env_cfg.spin_reward_axis_weights = tuple(spin_cfg["reward_axis_weights"])
        curriculum_cfg = spin_cfg.get("curriculum", {})
        if curriculum_cfg:
            env_cfg.spin_curriculum_enabled = curriculum_cfg.get(
                "enabled", env_cfg.spin_curriculum_enabled
            )
            env_cfg.spin_curriculum_initial_scale = curriculum_cfg.get(
                "initial_scale", env_cfg.spin_curriculum_initial_scale
            )
            env_cfg.spin_curriculum_full_fraction = curriculum_cfg.get(
                "full_fraction", env_cfg.spin_curriculum_full_fraction
            )
        shaping_cfg = spin_cfg.get("shaping", {})
        if shaping_cfg:
            env_cfg.shaping_final_scale = shaping_cfg.get(
                "final_scale", env_cfg.shaping_final_scale
            )
            env_cfg.shaping_full_fraction = shaping_cfg.get(
                "full_fraction", env_cfg.shaping_full_fraction
            )
        planner_cfg = spin_cfg.get("planner", {})
        if planner_cfg:
            env_cfg.planner_wrist_fraction = planner_cfg.get(
                "wrist_fraction", env_cfg.planner_wrist_fraction
            )
            env_cfg.planner_max_paddle_speed = planner_cfg.get(
                "max_paddle_speed", env_cfg.planner_max_paddle_speed
            )
            env_cfg.planner_max_paddle_angular_speed = planner_cfg.get(
                "max_paddle_angular_speed", env_cfg.planner_max_paddle_angular_speed
            )
            env_cfg.spin_planner_root_iterations = planner_cfg.get(
                "event_root_iterations", env_cfg.spin_planner_root_iterations
            )
            env_cfg.spin_planner_integration_substeps = planner_cfg.get(
                "event_integration_substeps",
                env_cfg.spin_planner_integration_substeps,
            )
        env_cfg.weighted_reward_keys["spin_target"] = spin_cfg.get(
            "reward_weight", env_cfg.weighted_reward_keys["spin_target"]
        )
        env_cfg.weighted_reward_keys["landing_target"] = spin_cfg.get(
            "landing_reward_weight", env_cfg.weighted_reward_keys["landing_target"]
        )
    return env_cfg


def run_train(config: dict, log_dir: str) -> None:
    """Training entry point. Video recording is only performed in the rank-0 process."""
    from on_policy_runner import OnPolicyRunner

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "":
        device = "cpu"
        local_rank = 0
        rank = 0
        world_size = 1
        seed = config["seed"]
        physical_gpu_id = None
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        print(f"local_rank: {local_rank}, rank: {rank}")

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
        device = f"cuda:{local_rank}"
        seed = config["seed"] + local_rank  # different seed per process for diversity

        cuda_visible_list = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
        physical_gpu_id = cuda_visible_list[local_rank] if local_rank < len(cuda_visible_list) else local_rank

    if cuda_visible != "":
        print(f"[INFO] Process Info: LOCAL_RANK={local_rank}, RANK={rank}, WORLD_SIZE={world_size}")
        print(f"[INFO] GPU Mapping: local_rank {local_rank} -> physical GPU {physical_gpu_id} -> device {device}")
        print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")
    else:
        print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")
    
    seed_rng(seed)

    env_cfg = apply_environment_config(tabletennis_p2_cfg(), config)
    env = TableTennisWarpEnv(env_cfg, device=device)
    env.reset()

    eval_env_cfg = apply_environment_config(tabletennis_p2_cfg(), config)
    eval_env_cfg.eval_env = True
    eval_env_cfg.num_envs = 1
    eval_env_cfg.nconmax = 200
    eval_env_cfg.njmax = 300
    eval_env_cfg.enable_domain_randomization = False
    eval_env_cfg.enable_action_randomization = False
    eval_env_cfg.render_height = 480
    eval_env_cfg.render_width = 640
    eval_env = TableTennisWarpEnv(eval_env_cfg, device=device)
    eval_env.reset()
    
    runner = OnPolicyRunner(
        env=env,
        eval_env=eval_env,
        train_cfg=config,
        log_dir=log_dir,
        device=device,
    )
    if config["resume_path"]:
        runner.load(config["resume_path"], load_optimizer=False)
        
    print(f"[DEBUG] Rank {rank} about to call learn()", flush=True)
    runner.learn(num_learning_iterations=config["max_iterations"])
    print(f"[DEBUG] Rank {rank} learn() completed", flush=True)

def launch_training(config: dict, gpu_ids: list[int] | Literal["all"] | None = None, log_dir: str = "logs/single_table_tennis/"):
    """
    Launch training with single-GPU or multi-GPU support.

    Args:
        config: Training configuration dict.
        gpu_ids: GPU selection - a list of GPU indices (e.g. [0, 1]),
                 "all" to use all available GPUs, or None for CPU / default single GPU.
    """
    log_dir = log_dir + datetime.now().strftime("%m%d%H%M") + "_seed" + str(config["seed"]) + "_lr" + str(config["algorithm"]["learning_rate"])

    selected_gpus, num_gpus = select_gpus(gpu_ids)

    if selected_gpus is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
    # os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_GL"] = "osmesa"

    
    if num_gpus <= 1:
        run_train(config, log_dir)
    else:
        import torchrunx

        logging.basicConfig(level=logging.INFO)

        torchrunx_log_dir = os.path.join(log_dir, "torchrunx")
        os.environ["TORCHRUNX_LOG_DIR"] = torchrunx_log_dir

        print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
        print(f"[INFO] Selected physical GPUs: {selected_gpus}", flush=True)
        print(f"[INFO] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
        print(f"[INFO] Each process will use: local_rank 0 -> GPU {selected_gpus[0]}, local_rank 1 -> GPU {selected_gpus[1] if len(selected_gpus) > 1 else 'N/A'}, ...", flush=True)
        torchrunx.Launcher(
            hostnames=["localhost"],
            workers_per_host=num_gpus,
            backend=None,
            copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
        ).run(run_train, config, log_dir)


def evaluate(config, path: str):
    from on_policy_runner import OnPolicyRunner

    seed_rng(config["seed"])

    env_cfg = apply_environment_config(tabletennis_p2_cfg(), config)
    env = TableTennisWarpEnv(env_cfg)
    env.reset()

    eval_env_cfg = apply_environment_config(tabletennis_p2_cfg(), config)
    eval_env_cfg.eval_env = True
    eval_env_cfg.num_envs = 1
    eval_env_cfg.nconmax = 200
    eval_env_cfg.njmax = 300
    eval_env_cfg.enable_domain_randomization = False
    eval_env_cfg.enable_action_randomization = False
    eval_env_cfg.render_height = 480
    eval_env_cfg.render_width = 640
    eval_env = TableTennisWarpEnv(eval_env_cfg)
    eval_env.reset()


    runner = OnPolicyRunner(
        env=env,
        eval_env=eval_env,
        train_cfg=config,
        log_dir="logs/single_tennis/" + datetime.now().strftime("%m%d%H%M"),
        device="cuda:0",
    )
    runner.evaluate_vis(path)

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Train double tennis agent")
    parser.add_argument(
        "--gpu-ids",
        type=str,
        nargs="+",
        default=["0"],
        help="GPU IDs to use for training. Use --gpu-ids 0 1 for multi-GPU, or --gpu-ids all for all GPUs",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/single_table_tennis/",
        help="Path to log directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for training",
    )
    parser.add_argument(
        "--action-type",
        type=str,
        default="joint_pd",
        help="Action type for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate for training",
    )
    
    args = parser.parse_args()
    
    if len(args.gpu_ids) == 1 and args.gpu_ids[0].lower() == "all":
        gpu_ids = "all"
    elif len(args.gpu_ids) == 1 and args.gpu_ids[0].lower() == "none":
        gpu_ids = None
    else:
        try:
            gpu_ids = [int(x) for x in args.gpu_ids]
        except ValueError:
            print(f"Error: Invalid GPU IDs: {args.gpu_ids}")
            print("Use --gpu-ids 0 1 for multi-GPU, --gpu-ids all for all GPUs, or --gpu-ids none for CPU")
            sys.exit(1)
    
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    config["seed"] = args.seed
    config["action_type"] = args.action_type
    if args.learning_rate is not None:
        config["algorithm"]["learning_rate"] = args.learning_rate

    launch_training(config, gpu_ids=gpu_ids, log_dir=args.log_dir)
    # evaluate(config, "/home/zwt/Projects/mujoco_wrap/mjlab/logs/competitive_tennis/11302142/model_7000.pt")
