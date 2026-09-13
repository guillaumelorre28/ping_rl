import time
from dataclasses import replace

import mujoco
import mujoco.viewer
import mujoco_warp as mjw
import numpy as np
import torch
from ball_physics import (
    DEFAULT_BALL_PHYSICS,
    aerodynamic_force_components,
    propagate_to_x,
    racket_impact,
    table_impact,
    trajectory_spin_to_world,
    world_spin_to_trajectory,
)
from ml_collections import config_dict
from muscle_utils import (
    calculate_vae_muscle_act,
    get_target_actuator_length,
    target_length_to_activations,
)
from planner import (
    DEFAULT_BALL_CONTACT_HEIGHT,
    MIN_RETURN_NET_HEIGHT,
    MIN_SERVE_NET_HEIGHT,
    compute_hit_pos,
    compute_land,
    compute_land_net,
    compute_paddle_pos,
    compute_paddle_vel,
    plan_spin_return,
    predict_incoming_hit,
    predict_land,
    vec_to_quat,
)
from rsl_rl.env import VecEnv
from tqdm import tqdm

from mjlab.sim.sim import Simulation, SimulationCfg

# from mjlab.third_party.isaaclab.isaaclab.utils import math
from mjlab.utils.lab_api import math


def recursive_immobilize(spec, temp_model, parent, remove_eqs=False, remove_actuators=False):
    removed_joint_ids = []
    for s in parent.sites:
        spec.delete(s)
    for j in parent.joints:
        removed_joint_ids.extend(temp_model.joint(j.name).qposadr)
        if remove_eqs:
            for e in spec.equalities:
                if e.type == mujoco.mjtEq.mjEQ_JOINT and (e.name1 == j.name or e.name2 == j.name):
                    spec.delete(e)
        if remove_actuators:
            for a in spec.actuators:
                if a.trntype == mujoco.mjtTrn.mjTRN_JOINT and a.target == j.name:
                    spec.delete(a)
        spec.delete(j)
    for child in parent.bodies:
        removed_joint_ids.extend(
            recursive_immobilize(spec, temp_model, child, remove_eqs, remove_actuators)
        )
    return removed_joint_ids


def recursive_remove_contacts(parent, return_condition=None):
    if return_condition is not None and return_condition(parent):
        return
    for g in parent.geoms:
        g.contype=0
        g.conaffinity=0
    for child in parent.bodies:
        recursive_remove_contacts(child, return_condition)


def recursive_mirror(meshes_to_mirror, spec_copy, parent):
    parent.pos[1] *= -1
    parent.quat[[1, 3]] *= -1
    parent.name += "_mirrored"

    for j in parent.joints:
        if j.name:
            j.name += "_mirrored"

    for g in parent.geoms:
        if g.type != mujoco.mjtGeom.mjGEOM_MESH:
            spec_copy.delete(g)
            continue
        g.pos[1] *= -1
        g.quat[[1, 3]] *= -1
        g.name += "_mirrored"
        g.group = 1
        meshes_to_mirror.add(g.meshname)
        g.meshname += "_mirrored"
    for child in parent.bodies:
        if "ping_pong" in child.name:
            spec_copy.detach_body(child)
            continue
        recursive_mirror(meshes_to_mirror, spec_copy, child)


def _set_multiccd(spec: mujoco.MjSpec, enabled: bool) -> None:
    """Positionne le flag multi-contact CCD, quelle que soit sa forme.

    MuJoCo a déplacé ce réglage d'un *enable* (``mjENBL_MULTICCD``, jusqu'en
    3.4) vers un *disable* (``mjDSBL_MULTICCD``, à partir de 3.13), et la
    révision de MuJoCo-Warp épinglée ici ne connaît ni l'un ni l'autre : elle
    refuse ``put_model`` avec un ``NotImplementedError`` sur le bit inconnu,
    dans les deux sens. On efface donc le flag sous les deux orthographes,
    sans supposer laquelle existe — c'est ce qui permet au même code de
    tourner sur le venv de développement et dans l'image.

    ``enabled=True`` laisse le flag tel que le XML l'a posé : le réglage reste
    accessible pour une révision de MuJoCo-Warp qui le supporterait, mais avec
    celle qui est épinglée, la construction de l'environnement échouera.
    """

    if enabled:
        return
    enable_bit = getattr(mujoco.mjtEnableBit, "mjENBL_MULTICCD", None)
    if enable_bit is not None:
        spec.option.enableflags &= ~int(enable_bit)
    disable_bit = getattr(mujoco.mjtDisableBit, "mjDSBL_MULTICCD", None)
    if disable_bit is not None:
        spec.option.disableflags &= ~int(disable_bit)


def apply_ball_inertia(mj_model: mujoco.MjModel) -> None:
    """Apply physical ball inertia after compilation without changing global bounds."""

    ball_id = mj_model.body("pingpong").id
    mj_model.body_inertia[ball_id, :] = DEFAULT_BALL_PHYSICS.shell_inertia
    # body_inertia contributes to derived fields such as dof_invweight0.
    mujoco.mj_setConst(mj_model, mujoco.MjData(mj_model))


def apply_spin_ablation_config(cfg: config_dict.ConfigDict) -> config_dict.ConfigDict:
    """Restore the pre-spin observation and reward interfaces when disabled."""

    if cfg.spin_physics_enabled:
        return cfg
    spin_observations = {
        "ball_spin",
        "paddle_angvel",
        "target_spin",
        "target_paddle_angvel",
        "ball_physics_params",
    }
    cfg.obs_keys = [key for key in cfg.obs_keys if key not in spin_observations]
    cfg.critic_obs_keys = [
        key for key in cfg.critic_obs_keys if key not in spin_observations
    ]
    legacy_reward_keys = {
        "rel_pos_err",
        "rel_quat_err",
        "fin_open",
        "paddle_pos_err",
        "paddle_ori_err",
        "hit_with_paddle",
        "fall_opponent",
        "fall_plane_dist",
        "fall_hit_plane",
        "net_penalty",
        "act_reg",
    }
    cfg.weighted_reward_keys = {
        key: value
        for key, value in cfg.weighted_reward_keys.items()
        if key in legacy_reward_keys
    }
    return cfg


def tabletennis_p2_cfg():
    return config_dict.create(
        # simulation configs
        model_path="tabletennis.xml",
        num_envs=1024,
        eval_env=False,
        nconmax=50000,
        njmax=200,
        frame_skip=5,
        enable_multiccd=False,
        action_type="joint_pd",  # choose from joint_pd, muscle_pd, muscle_act, muscle_vae
        kp_scale=10.0,
        kd_scale=0.1,
        # muscle_vae specific configs
        kp_vae=10.0,
        kd_vae=1.0,
        # task configs
        max_episode_length=300,
        normalize_act=True,
        ball_qvel=True,
        paddle_mass_range=(0.10, 0.15),
        # paddle_mass_range=(0.05, 0.25),
        # -0.9634, -0.2302, 1.4041
        ball_xyz_range=config_dict.create(
            high=(-1.79, 0.7, 1.35),
            low=(-1.81, -0.7, 1.1),
        ),
        hit_xyz_range=config_dict.create(
            high=(1.81, 0.7, 1.30),
            low=(1.79, -0.7, 1.00),
        ),
        ball_friction_range=config_dict.create(
            high=(1.1, 0.006, 0.00003),
            low=(0.9, 0.004, 0.00001),
        ),
        ball_limited_range=config_dict.create(
            high=(2.5, 1.5, 2.5),
            low=(-2.0, -1.5, 0.795),
        ),
        # Spin-aware ball physics. Commands use trajectory coordinates:
        # (topspin, sidespin, corkscrew), in rad/s.
        spin_physics_enabled=True,
        analytic_contact_override=True,
        reciprocal_contact_impulse=True,
        aerodynamic_model="conti2026",  # conti2026 or constant
        drag_enabled=True,
        magnus_enabled=True,
        incoming_spin_enabled=True,
        spin_planner_enabled=True,
        spin_reward_enabled=True,
        spin_planner_root_iterations=4,
        spin_planner_integration_substeps=3,
        table_contact_height=DEFAULT_BALL_CONTACT_HEIGHT,
        spin_command_mode="uniform",  # fixed or uniform
        spin_target=(0.0, 0.0, 0.0),
        spin_target_range=config_dict.create(
            low=(-400.0, -250.0, 0.0),
            high=(400.0, 250.0, 0.0),
        ),
        incoming_spin_range=config_dict.create(
            low=(-350.0, -250.0, -100.0),
            high=(350.0, 250.0, 100.0),
        ),
        spin_reward_sigma=(80.0, 80.0, 100.0),
        spin_reward_axis_weights=(1.0, 1.0, 0.0),
        spin_priority=1.0,
        spin_curriculum_enabled=True,
        spin_curriculum_initial_scale=0.25,
        spin_curriculum_full_fraction=0.40,
        shaping_final_scale=0.20,
        shaping_full_fraction=0.60,
        planner_wrist_fraction=0.35,
        planner_max_paddle_speed=12.0,
        planner_max_paddle_angular_speed=30.0,
        contact_release_substeps=2,
        drag_scale_range=(0.90, 1.10),
        magnus_scale_range=(0.85, 1.15),
        table_friction_physics_range=(0.22, 0.28),
        racket_grip_scale_range=(0.90, 1.10),
        obs_keys=[
            # "time",
            "pelvis_pos",
            "body_qpos",
            "body_qvel",
            "ball_pos",
            "ball_vel",
            "ball_spin",
            "paddle_pos",
            "paddle_vel",
            "paddle_angvel",
            "paddle_ori",
            "reach_err",
            "touching_info",
            "act",
            "target_pos",
            "target_vel",
            "target_spin",
            "target_paddle_angvel",
            # "target_time",
        ],
        critic_obs_keys=[
            "time",
            "pelvis_pos",
            "body_qpos",
            "body_qvel",
            "ball_pos",
            "ball_vel",
            "ball_spin",
            "paddle_pos",
            "paddle_vel",
            "paddle_angvel",
            "paddle_ori",
            "reach_err",
            "touching_info",
            "act",
            "actuator_length",
            "actuator_velocity",
            "target_pos",
            "target_vel",
            "target_spin",
            "target_paddle_angvel",
            "target_time",
            "paddle_mass",
            "ball_friction",
            "ball_physics_params",
        ],
        weighted_reward_keys={
            "rel_pos_err": 4,
            "rel_quat_err": 4,
            "fin_open": 10,
            "paddle_pos_err": 20,
            "paddle_ori_err": 10,
            "paddle_contact_vel_err": 5,
            "hit_with_paddle": 100,
            "spin_target": 60.0,
            "spin_error": 0,
            "fall_opponent": 100,
            "fall_plane_dist": 100,
            "landing_target": 40.0,
            "landing_error": 0,
            "fall_hit_plane": 100,
            "net_penalty": -20,
            "act_reg": 0,
        },
        # domain randomization
        enable_domain_randomization=False,
        pos_jitter_range=0.02,
        vel_jitter_range=0.2,
        spin_jitter_range=20.0,
        paddle_angvel_jitter_range=1.0,
        act_jitter_range=0.05,
        enable_action_randomization=True,
        action_range=0.05,
        act_range=0.1,
    )


class TableTennisWarpEnv(VecEnv):
    @staticmethod
    def _preprocess_spec(
        spec: mujoco.MjSpec,
        remove_body_collisions: bool = True,
        add_left_arm: bool = True,
    ) -> mujoco.MjSpec:
        """Preprocess the MuJoCo spec to:
        - Immobilize leg joints
        - Remove unnecessary body collisions
        - Optionally add mirrored left arm
        
        Args:
            spec: The MuJoCo spec to preprocess
            remove_body_collisions: Whether to remove body collisions
            add_left_arm: Whether to add mirrored left arm
            
        Returns:
            The preprocessed spec
        """
        for s in spec.sensors:
            if "pingpong" not in s.name and "paddle" not in s.name and "ball" not in s.name:
                spec.delete(s)
        # Compile a temporary model to get joint information
        temp_model = spec.compile()
        
        # Immobilize leg joints
        removed_ids = recursive_immobilize(spec, temp_model, spec.body("femur_l"), remove_eqs=True)
        removed_ids.extend(recursive_immobilize(spec, temp_model, spec.body("femur_r"), remove_eqs=True))


        for key in spec.keys:
            key.qpos = [j for i, j in enumerate(key.qpos) if i not in removed_ids]

        if remove_body_collisions:
            recursive_remove_contacts(spec.body("full_body"), return_condition=lambda b: "radius" in b.name)
        
        return spec
    
    def __init__(self, cfg: config_dict.ConfigDict, device: str = "cuda:0"):
        cfg = apply_spin_ablation_config(cfg)
        self.cfg = cfg
        self.ball_physics = replace(
            DEFAULT_BALL_PHYSICS,
            aerodynamic_model=cfg.aerodynamic_model,
            drag_enabled=cfg.drag_enabled,
            magnus_enabled=cfg.magnus_enabled,
        )
        # Load spec and preprocess it
        spec: mujoco.MjSpec = mujoco.MjSpec.from_file(cfg.model_path)
        if not cfg.spin_physics_enabled:
            for pair in list(spec.pairs):
                if pair.geomname1 == "pingpong" and pair.geomname2 in {
                    "coll_own_half",
                    "coll_opponent_half",
                    "pad",
                }:
                    spec.delete(pair)
        _set_multiccd(spec, cfg.enable_multiccd)
        # spec = self._preprocess_spec(spec, remove_body_collisions=True, add_left_arm=True)
        self.mj_model = spec.compile()
        if cfg.spin_physics_enabled:
            apply_ball_inertia(self.mj_model)
        self.num_envs = cfg.num_envs
        self.max_episode_length = cfg.max_episode_length
        self.device = torch.device(device)
        # sim_cfg = SimulationCfg(
        #     nconmax=cfg.nconmax,
        #     njmax=cfg.njmax,
        #     mujoco=MujocoCfg(integrator="euler"),
        # )
        sim_cfg = SimulationCfg(nconmax=100)
        self.sim = Simulation(num_envs=self.num_envs, cfg=sim_cfg, model=self.mj_model, device=device)
        # domain randomization
        self.sim.expand_model_fields(["body_mass", "geom_friction", "body_pos"])
        self.sim.create_graph()
        self.renderer = None

        self.episode_length_buf = torch.zeros(self.num_envs, device=device, dtype=torch.long)
        self.touching_info = torch.zeros(self.num_envs, 6, device=device, dtype=torch.bool)
        # ball touching state: 0 - start, 1 - touching the paddle, 2 - leave the paddle, 3 - touching something after leaving the paddle
        self.touching_state = torch.zeros(self.num_envs, device=device, dtype=torch.int)
        self.after_leaving_paddle = torch.zeros(self.num_envs, device=device, dtype=torch.int)
        # ball landing state: 0 - not landing, 1 - landing, 2 - leave the table
        self.landing_state = torch.zeros(self.num_envs, device=device, dtype=torch.int)
        self.hit_with_paddle_count = torch.zeros(self.num_envs, device=device, dtype=torch.int)

        self.after_leaving_own = torch.zeros(self.num_envs, device=device, dtype=torch.int)

        self.drag_scale = torch.ones(self.num_envs, device=device)
        self.magnus_scale = torch.ones(self.num_envs, device=device)
        self.table_friction_physics = torch.full(
            (self.num_envs,), DEFAULT_BALL_PHYSICS.table_friction, device=device
        )
        self.racket_grip_scale = torch.ones(self.num_envs, device=device)
        self._substep_contact_info = torch.zeros(self.num_envs, 6, device=device, dtype=torch.bool)
        self._contact_latched = torch.zeros(self.num_envs, 6, device=device, dtype=torch.bool)
        self._contact_separation_steps = torch.zeros(
            self.num_envs, 6, device=device, dtype=torch.int
        )
        self.post_hit_ball_vel = torch.zeros(self.num_envs, 3, device=device)
        self.post_hit_ball_spin = torch.zeros(self.num_envs, 3, device=device)
        self.has_post_hit_state = torch.zeros(self.num_envs, device=device, dtype=torch.bool)

        self.target_pos = torch.zeros(self.num_envs, 3, device=device)
        self.target_vel = torch.zeros(self.num_envs, 3, device=device)
        self.target_paddle_angvel = torch.zeros(self.num_envs, 3, device=device)
        self.target_contact_vel = torch.zeros(self.num_envs, 3, device=device)
        self.target_contact_offset_local = torch.zeros(self.num_envs, 3, device=device)
        self.target_ori = torch.zeros(self.num_envs, 4, device=device)
        self.target_spin = torch.zeros(self.num_envs, 3, device=device)
        self.target_spin_world = torch.zeros(self.num_envs, 3, device=device)
        self.target_landing_pos = torch.zeros(self.num_envs, 3, device=device)
        self.target_plan_valid = torch.zeros(self.num_envs, device=device, dtype=torch.bool)
        self.hit_pos = torch.zeros(self.num_envs, 3, device=device)
        self.target_time = torch.zeros(self.num_envs, device=device, dtype=torch.float)
        self.target_time_tolerance = 0.05
        self.leaving_paddle_tolerance = 3
        self.leaving_table_tolerance = 1

        self.equalities = []
        self.constrained_joints = []
        self.action_joints = []

        self.last_obs = None
        self.current_obs = None
        self.extras = None

        self.current_episode_length = 0
        self.hit_step = torch.zeros(self.num_envs, 1, device=device)
        self.paddle_face_dir_local = torch.zeros(1, 3, device=device)
        self.paddle_face_dir_local[0,2] = -1

        self.opponent_table_upper = torch.tensor([-1.35, 0.50, self.cfg.table_contact_height]).to(self.device)
        self.opponent_table_lower = torch.tensor([-0.5, -0.40, self.cfg.table_contact_height]).to(self.device)

        for i in range(self.mj_model.neq):
            self.equalities.append([self.mj_model.eq_obj1id[i], self.mj_model.eq_obj2id[i], self.mj_model.eq_data[i]])
            self.constrained_joints.append(self.mj_model.eq_obj1id[i])

        for i in range(self.mj_model.njnt):
            if (
                self.mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
                or self.mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_SLIDE
            ) and i not in self.constrained_joints:
                self.action_joints.append(i)

        if self.cfg.action_type == "joint_pd":
            self.num_actions = len(self.action_joints)
        elif self.cfg.action_type in ["muscle_pd", "muscle_act", "muscle_vae"]:
            self.num_actions = self.sim.data.ctrl.shape[1]

        self.action_joints = torch.tensor(self.action_joints).to(device)
        self.action_low = torch.from_numpy(self.mj_model.jnt_range[:, 0]).to(device)[self.action_joints].float()
        self.action_high = torch.from_numpy(self.mj_model.jnt_range[:, 1]).to(device)[self.action_joints].float()
        self.action_to_qpos = torch.from_numpy(self.mj_model.jnt_qposadr).to(device)[self.action_joints]

        self._post_init()

    @property
    def model(self) -> mjw.Model:
        return self.sim.model

    @property
    def data(self) -> mjw.Data:
        return self.sim.data

    @property
    def fk_data(self) -> mjw.Data:
        return self.sim.fk_data

    def _post_init(self):
        """record some index to calculate observations and rewards"""
        self.muscle_ind = torch.from_numpy(self.mj_model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE).to(self.device)
        self.non_muscle_ind = torch.from_numpy(self.mj_model.actuator_dyntype != mujoco.mjtDyn.mjDYN_MUSCLE).to(
            self.device
        )
        self.non_muscle_low = (
            torch.from_numpy(
                self.mj_model.actuator_ctrlrange[self.mj_model.actuator_dyntype != mujoco.mjtDyn.mjDYN_MUSCLE, 0]
            )
            .to(self.device)
            .float()
        )
        self.non_muscle_high = (
            torch.from_numpy(
                self.mj_model.actuator_ctrlrange[self.mj_model.actuator_dyntype != mujoco.mjtDyn.mjDYN_MUSCLE, 1]
            )
            .to(self.device)
            .float()
        )

        self.init_qpos = torch.from_numpy(self.mj_model.key_qpos[0].copy()).to(self.device).float()

        self.ball_xyz_low = torch.tensor(self.cfg.ball_xyz_range.low).to(self.device).float()
        self.ball_xyz_high = torch.tensor(self.cfg.ball_xyz_range.high).to(self.device).float()
        self.hit_xyz_low = torch.tensor(self.cfg.hit_xyz_range.low).to(self.device).float()
        self.hit_xyz_high = torch.tensor(self.cfg.hit_xyz_range.high).to(self.device).float()
        self.ball_friction_low = torch.tensor(self.cfg.ball_friction_range.low).to(self.device).float()
        self.ball_friction_high = torch.tensor(self.cfg.ball_friction_range.high).to(self.device).float()
        self.paddle_mass_low = self.cfg.paddle_mass_range[0]
        self.paddle_mass_high = self.cfg.paddle_mass_range[1]
        self.ball_limited_low = torch.tensor(self.cfg.ball_limited_range.low).to(self.device).float()
        self.ball_limited_high = torch.tensor(self.cfg.ball_limited_range.high).to(self.device).float()
        self.spin_target_low = torch.tensor(self.cfg.spin_target_range.low, device=self.device).float()
        self.spin_target_high = torch.tensor(self.cfg.spin_target_range.high, device=self.device).float()
        self.incoming_spin_low = torch.tensor(self.cfg.incoming_spin_range.low, device=self.device).float()
        self.incoming_spin_high = torch.tensor(self.cfg.incoming_spin_range.high, device=self.device).float()
        self.spin_reward_sigma = torch.tensor(self.cfg.spin_reward_sigma, device=self.device).float()
        self.spin_reward_axis_weights = torch.tensor(
            self.cfg.spin_reward_axis_weights, device=self.device
        ).float()

        self.opponent_center = torch.tensor([-0.85, 0.04, 0.795]).to(self.device)
        self.plane_center = (self.ball_xyz_low + self.ball_xyz_high) / 2

        self.palm_sid = self.mj_model.site("S_grasp").id
        self.fin0_sid = self.mj_model.site("THtip").id
        self.fin1_sid = self.mj_model.site("IFtip").id
        self.fin2_sid = self.mj_model.site("MFtip").id
        self.fin3_sid = self.mj_model.site("RFtip").id
        self.fin4_sid = self.mj_model.site("LFtip").id

        self.pelvis_sid = self.mj_model.site("pelvis").id
        self.paddle_sid = self.mj_model.site("paddle").id
        self.paddle_bid = self.mj_model.body("paddle").id
        self.ball_sid = self.mj_model.site("pingpong").id
        self.ball_bid = self.mj_model.body("pingpong").id
        self.grasp_sid = self.mj_model.site("S_grasp").id
        self.paddle_site_pos_local = torch.from_numpy(
            self.mj_model.site_pos[self.paddle_sid].copy()
        ).to(self.device).float()
        self.paddle_inertia = torch.from_numpy(
            self.mj_model.body_inertia[self.paddle_bid].copy()
        ).to(self.device).float()
        self.paddle_inertial_quat = torch.from_numpy(
            self.mj_model.body_iquat[self.paddle_bid].copy()
        ).to(self.device).float()

        self.ball_bid = self.mj_model.body("pingpong").id
        self.ball_gid = self.mj_model.geom("pingpong").id
        self.own_half_gid = self.mj_model.geom("coll_own_half").id
        self.paddle_gid = self.mj_model.geom("pad").id
        self.opponent_half_gid = self.mj_model.geom("coll_opponent_half").id
        self.ground_gid = self.mj_model.geom("ground").id
        self.net_gid = self.mj_model.geom("coll_net").id

        self.ball_sensor_adr = self.mj_model.sensor("pingpong_vel_sensor").adr[0]
        self.ball_sensor_dim = self.mj_model.sensor("pingpong_vel_sensor").dim[0]
        self.ball_angvel_sensor_adr = self.mj_model.sensor("pingpong_angvel_sensor").adr[0]
        self.ball_angvel_sensor_dim = self.mj_model.sensor("pingpong_angvel_sensor").dim[0]
        self.paddle_sensor_adr = self.mj_model.sensor("paddle_vel_sensor").adr[0]
        self.paddle_sensor_dim = self.mj_model.sensor("paddle_vel_sensor").dim[0]
        self.paddle_angvel_sensor_adr = self.mj_model.sensor("paddle_angvel_sensor").adr[0]
        self.paddle_angvel_sensor_dim = self.mj_model.sensor("paddle_angvel_sensor").dim[0]

        self.ball_paddle_sensor_adr = self.mj_model.sensor("ball_paddle_contact").adr[0]
        self.ball_paddle_sensor_dim = self.mj_model.sensor("ball_paddle_contact").dim[0]
        self.ball_own_sensor_adr = self.mj_model.sensor("ball_own_contact").adr[0]
        self.ball_own_sensor_dim = self.mj_model.sensor("ball_own_contact").dim[0]
        self.ball_opponent_sensor_adr = self.mj_model.sensor("ball_opponent_contact").adr[0]
        self.ball_opponent_sensor_dim = self.mj_model.sensor("ball_opponent_contact").dim[0]
        self.ball_ground_sensor_adr = self.mj_model.sensor("ball_ground_contact").adr[0]
        self.ball_ground_sensor_dim = self.mj_model.sensor("ball_ground_contact").dim[0]
        self.ball_net_sensor_adr = self.mj_model.sensor("ball_net_contact").adr[0]
        self.ball_net_sensor_dim = self.mj_model.sensor("ball_net_contact").dim[0]
        self.ball_other_sensor_adr = self.mj_model.sensor("ball_other_contact").adr[0]
        self.ball_other_sensor_dim = self.mj_model.sensor("ball_other_contact").dim[0]

        self.ball_dofadr = self.mj_model.body_dofadr[self.ball_bid]
        self.ball_posadr = self.mj_model.joint("pingpong_freejoint").qposadr[0]
        self.paddle_dofadr = self.mj_model.joint("paddle_freejoint").dofadr[0]
        self.paddle_posadr = self.mj_model.joint("paddle_freejoint").qposadr[0]

        myo_bodies = [
            self.mj_model.body(i).id
            for i in range(self.mj_model.nbody)
            if not self.mj_model.body(i).name.startswith("ping")
            and "paddle" not in self.mj_model.body(i).name
            and self.mj_model.body(i).name not in ["pingpong"]
        ]
        self.myo_body_range = (min(myo_bodies), max(myo_bodies))

        self.myo_joint_range = np.concatenate(
            [
                self.mj_model.joint(i).qposadr
                for i in range(self.mj_model.njnt)
                if not self.mj_model.joint(i).name.startswith("ping")
                and not self.mj_model.joint(i).name == "pingpong_freejoint"
                and not self.mj_model.joint(i).name == "paddle_freejoint"
            ]
        )

        self.myo_dof_range = np.concatenate(
            [
                self.mj_model.joint(i).dofadr
                for i in range(self.mj_model.njnt)
                if not self.mj_model.joint(i).name.startswith("ping")
                and not self.mj_model.joint(i).name == "paddle_freejoint"
            ]
        )

        if self.cfg.action_type == "muscle_vae":
            self._compute_tpose_muscle_length()

    def _compute_tpose_muscle_length(self):
        """Compute muscle lengths at the t-pose (initial keyframe) for muscle_vae action mapping."""
        self.fk_data.qpos[:] = self.init_qpos.unsqueeze(0).repeat(self.num_envs, 1)
        self.sim.fk_forward()
        # All envs share the same initial pose; use env 0 as reference
        self.tpose_muscle_length = self.fk_data.actuator_length[0].clone()
        self.tpose_muscle_length_muscles = self.tpose_muscle_length[self.muscle_ind].clone()

    def _read_touching_info(self) -> torch.Tensor:
        """Read all ball contact sensors without changing episode state."""

        paddle_contact = (
            self.data.sensordata[
                :, self.ball_paddle_sensor_adr : self.ball_paddle_sensor_adr + self.ball_paddle_sensor_dim
            ]
            > 0
        )
        own_contact = (
            self.data.sensordata[:, self.ball_own_sensor_adr : self.ball_own_sensor_adr + self.ball_own_sensor_dim] > 0
        )
        opponent_contact = (
            self.data.sensordata[
                :, self.ball_opponent_sensor_adr : self.ball_opponent_sensor_adr + self.ball_opponent_sensor_dim
            ]
            > 0
        )
        ground_contact = (
            self.data.sensordata[
                :, self.ball_ground_sensor_adr : self.ball_ground_sensor_adr + self.ball_ground_sensor_dim
            ]
            > 0
        )
        net_contact = (
            self.data.sensordata[:, self.ball_net_sensor_adr : self.ball_net_sensor_adr + self.ball_net_sensor_dim] > 0
        )
        env_contact = (
            self.data.sensordata[
                :, self.ball_other_sensor_adr : self.ball_other_sensor_adr + self.ball_other_sensor_dim
            ]
            > 0
        )
        env_contact &= ~paddle_contact
        env_contact &= ~own_contact
        env_contact &= ~opponent_contact
        env_contact &= ~ground_contact
        env_contact &= ~net_contact
        return torch.cat(
            [paddle_contact, own_contact, opponent_contact, ground_contact, net_contact, env_contact], dim=-1
        )

    def _cal_touching_info(self, touching_info: torch.Tensor | None = None) -> torch.Tensor:
        """Update episode contact state, optionally from substep-aggregated contacts."""

        self.touching_info = self._read_touching_info() if touching_info is None else touching_info
        paddle_contact = self.touching_info[:, 0:1]
        own_contact = self.touching_info[:, 1:2]
        opponent_contact = self.touching_info[:, 2:3]
        ground_contact = self.touching_info[:, 3:4]
        net_contact = self.touching_info[:, 4:5]
        env_contact = self.touching_info[:, 5:6]

        # touching_state==0 & paddle_contact -> touching_state=1 touching the paddle
        self.touching_state = torch.where(
            (self.touching_state == 0) & paddle_contact.squeeze(-1), 1, self.touching_state
        )
        # touching_state==1 & ~paddle_contact -> touching_state=2 leaving the paddle
        self.touching_state = torch.where(
            (self.touching_state == 1) & ~paddle_contact.squeeze(-1), 2, self.touching_state
        )

        # touching_state==2 & (paddle_contact | own_contact | ground_contact | net_contact | env_contact) -> touching_state=3 touching other things after leaving the paddle
        self.touching_state = torch.where(
            (self.touching_state == 2)
            & (paddle_contact | own_contact | ground_contact | net_contact | env_contact).squeeze(-1),
            3,
            self.touching_state,
        )

        # touching_state==2 & opponent_contact -> touching_state=4 touching the opponent side
        self.touching_state = torch.where(
            (self.touching_state == 2)
            & (opponent_contact).squeeze(-1),
            4,
            self.touching_state,
        )

        self.after_leaving_paddle[self.touching_state >= 2] += 1 # after leaving the paddle, count the number of timesteps

        # landing_state==0 & own_contact -> landing_state=1
        self.landing_state = torch.where((self.landing_state == 0) & own_contact.squeeze(-1), 1, self.landing_state)
        # landing_state==1 & ~own_contact -> landing_state=2
        self.landing_state = torch.where((self.landing_state == 1) & ~own_contact.squeeze(-1), 2, self.landing_state)

        self.after_leaving_own[self.landing_state == 1] = 0
        self.after_leaving_own[self.landing_state >= 2] += 1

        return self.touching_info

    def _ball_kinematics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read the current ball state directly from qpos/qvel.

        MuJoCo sensor buffers describe the state at the last forward pass.  The
        analytic contact override modifies qvel after that pass, so direct
        reads prevent a one-control-step stale spin observation.
        """

        position = self.data.qpos[:, self.ball_posadr : self.ball_posadr + 3]
        orientation = self.data.qpos[:, self.ball_posadr + 3 : self.ball_posadr + 7]
        velocity = self.data.qvel[:, self.ball_dofadr : self.ball_dofadr + 3]
        local_spin = self.data.qvel[:, self.ball_dofadr + 3 : self.ball_dofadr + 6]
        return position, velocity, math.quat_apply(orientation, local_spin)

    def _paddle_kinematics(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return paddle site pose/velocity and body twist from qpos/qvel."""

        body_position = self.data.qpos[:, self.paddle_posadr : self.paddle_posadr + 3]
        orientation = self.data.qpos[:, self.paddle_posadr + 3 : self.paddle_posadr + 7]
        body_velocity = self.data.qvel[:, self.paddle_dofadr : self.paddle_dofadr + 3]
        local_angular_velocity = self.data.qvel[
            :, self.paddle_dofadr + 3 : self.paddle_dofadr + 6
        ]
        angular_velocity = math.quat_apply(orientation, local_angular_velocity)
        site_offset = math.quat_apply(
            orientation, self.paddle_site_pos_local.expand(self.num_envs, -1)
        )
        site_position = body_position + site_offset
        site_velocity = body_velocity + torch.cross(angular_velocity, site_offset, dim=-1)
        return site_position, site_velocity, angular_velocity, body_position, orientation

    def _update_contact_latches(self, contact_info: torch.Tensor) -> torch.Tensor:
        """Return debounced contact edges and update per-contact hysteresis."""

        rising = contact_info & ~self._contact_latched
        self._contact_separation_steps = torch.where(
            contact_info,
            torch.zeros_like(self._contact_separation_steps),
            self._contact_separation_steps + 1,
        )
        released = self._contact_separation_steps >= int(self.cfg.contact_release_substeps)
        self._contact_latched = torch.where(
            contact_info,
            torch.ones_like(self._contact_latched),
            self._contact_latched & ~released,
        )
        self._substep_contact_info = contact_info.clone()
        return rising

    def _apply_aerodynamics(self) -> None:
        """Apply drag and Magnus force to the ball before one physics substep."""

        if not self.cfg.spin_physics_enabled:
            self.data.xfrc_applied[:, self.ball_bid, :] = 0.0
            return

        _, ball_vel, ball_spin = self._ball_kinematics()
        drag, magnus = aerodynamic_force_components(ball_vel, ball_spin, self.ball_physics)
        force = self.drag_scale.unsqueeze(-1) * drag + self.magnus_scale.unsqueeze(-1) * magnus
        self.data.xfrc_applied[:, self.ball_bid, 0:3] = force
        self.data.xfrc_applied[:, self.ball_bid, 3:6] = 0.0

    def _apply_analytic_contacts(
        self,
        pre_ball_pos: torch.Tensor,
        pre_ball_vel: torch.Tensor,
        pre_ball_spin: torch.Tensor,
        pre_paddle_body_pos: torch.Tensor,
        pre_paddle_vel: torch.Tensor,
        pre_paddle_spin: torch.Tensor,
        pre_paddle_quat: torch.Tensor,
        rising: torch.Tensor,
    ) -> None:
        """Replace the first native-contact result with identified impact maps."""

        paddle_mask = rising[:, 0]
        table_mask = (rising[:, 1] | rising[:, 2]) & ~paddle_mask

        table_out_vel, table_out_spin, table_info = table_impact(
            pre_ball_vel,
            pre_ball_spin,
            params=self.ball_physics,
            friction=self.table_friction_physics,
        )
        table_mask &= table_info["incoming"]

        paddle_normal = math.quat_apply(
            pre_paddle_quat,
            self.paddle_face_dir_local.expand(self.num_envs, -1),
        )
        relative_center_velocity = pre_ball_vel - pre_paddle_vel
        normal_sign = torch.where(
            torch.sum(relative_center_velocity * paddle_normal, dim=-1, keepdim=True) > 0.0,
            -torch.ones_like(paddle_normal[..., :1]),
            torch.ones_like(paddle_normal[..., :1]),
        )
        paddle_normal = paddle_normal * normal_sign
        contact_point = pre_ball_pos - DEFAULT_BALL_PHYSICS.radius * paddle_normal
        contact_offset = contact_point - pre_paddle_body_pos
        paddle_out_vel, paddle_out_spin, paddle_info = racket_impact(
            pre_ball_vel,
            pre_ball_spin,
            pre_paddle_vel,
            pre_paddle_spin,
            paddle_normal,
            contact_offset,
            params=self.ball_physics,
            grip_scale=self.racket_grip_scale,
        )
        paddle_mask &= paddle_info["incoming"]

        current_vel = self.data.qvel[:, self.ball_dofadr : self.ball_dofadr + 3]
        current_ball_quat = self.data.qpos[
            :, self.ball_posadr + 3 : self.ball_posadr + 7
        ]
        current_spin_local = self.data.qvel[
            :, self.ball_dofadr + 3 : self.ball_dofadr + 6
        ]
        current_spin = math.quat_apply(current_ball_quat, current_spin_local)
        corrected_vel = torch.where(table_mask.unsqueeze(-1), table_out_vel, current_vel)
        corrected_spin = torch.where(table_mask.unsqueeze(-1), table_out_spin, current_spin)
        corrected_vel = torch.where(paddle_mask.unsqueeze(-1), paddle_out_vel, corrected_vel)
        corrected_spin = torch.where(paddle_mask.unsqueeze(-1), paddle_out_spin, corrected_spin)

        if self.cfg.reciprocal_contact_impulse:
            # The native solver has already transferred its impulse to the
            # constrained racket/arm.  Apply the opposite difference between
            # the analytic and native ball momenta, so replacing the ball state
            # does not inject net momentum into the hybrid system.
            delta_ball_momentum = self.ball_physics.mass * (paddle_out_vel - current_vel)
            delta_ball_momentum = torch.where(
                paddle_mask.unsqueeze(-1),
                delta_ball_momentum,
                torch.zeros_like(delta_ball_momentum),
            )
            paddle_mass = self.model.body_mass[:, self.paddle_bid].unsqueeze(-1).clamp_min(1.0e-5)
            self.data.qvel[:, self.paddle_dofadr : self.paddle_dofadr + 3] -= (
                delta_ball_momentum / paddle_mass
            )

            current_ball_position = self.data.qpos[
                :, self.ball_posadr : self.ball_posadr + 3
            ]
            current_paddle_position = self.data.qpos[
                :, self.paddle_posadr : self.paddle_posadr + 3
            ]
            lever = current_ball_position - current_paddle_position
            delta_ball_spin_momentum = self.ball_physics.shell_inertia * (
                paddle_out_spin - current_spin
            )
            delta_paddle_angular_momentum = -torch.cross(
                lever, delta_ball_momentum, dim=-1
            ) - torch.where(
                paddle_mask.unsqueeze(-1),
                delta_ball_spin_momentum,
                torch.zeros_like(delta_ball_spin_momentum),
            )
            paddle_quat = self.data.qpos[
                :, self.paddle_posadr + 3 : self.paddle_posadr + 7
            ]
            inertial_quat = math.quat_mul(
                paddle_quat,
                self.paddle_inertial_quat.expand(self.num_envs, -1),
            )
            angular_momentum_inertial = math.quat_apply_inverse(
                inertial_quat, delta_paddle_angular_momentum
            )
            angular_velocity_inertial = angular_momentum_inertial / self.paddle_inertia.clamp_min(
                1.0e-8
            )
            angular_velocity_world = math.quat_apply(
                inertial_quat, angular_velocity_inertial
            )
            self.data.qvel[
                :, self.paddle_dofadr + 3 : self.paddle_dofadr + 6
            ] += math.quat_apply_inverse(paddle_quat, angular_velocity_world)
        corrected_spin_local = math.quat_apply_inverse(current_ball_quat, corrected_spin)
        self.data.qvel[:, self.ball_dofadr : self.ball_dofadr + 3] = corrected_vel
        self.data.qvel[:, self.ball_dofadr + 3 : self.ball_dofadr + 6] = corrected_spin_local
        self.post_hit_ball_vel = torch.where(
            paddle_mask.unsqueeze(-1), paddle_out_vel, self.post_hit_ball_vel
        )
        self.post_hit_ball_spin = torch.where(
            paddle_mask.unsqueeze(-1), paddle_out_spin, self.post_hit_ball_spin
        )
        self.has_post_hit_state |= paddle_mask

    def _update_current_obs(self) -> None:
        sim_time = self.data.time
        pelvis_pos = self.data.site_xpos[:, self.pelvis_sid]
        noised_pelvis_pos = pelvis_pos + torch.randn_like(pelvis_pos) * self.cfg.pos_jitter_range

        body_qpos = self.data.qpos[:, self.myo_joint_range]
        body_qvel = self.data.qvel[:, self.myo_dof_range]
        noised_body_qpos = body_qpos + torch.randn_like(body_qpos) * self.cfg.pos_jitter_range
        noised_body_qvel = body_qvel + torch.randn_like(body_qvel) * self.cfg.vel_jitter_range

        ball_pos, ball_vel, ball_spin = self._ball_kinematics()
        paddle_pos, paddle_vel, paddle_angvel, _, paddle_ori = self._paddle_kinematics()
        noised_paddle_pos = paddle_pos + torch.randn_like(paddle_pos) * self.cfg.pos_jitter_range
        noised_paddle_vel = paddle_vel + torch.randn_like(paddle_vel) * self.cfg.vel_jitter_range
        noised_ball_spin = ball_spin + torch.randn_like(ball_spin) * self.cfg.spin_jitter_range
        noised_paddle_angvel = (
            paddle_angvel + torch.randn_like(paddle_angvel) * self.cfg.paddle_angvel_jitter_range
        )

        reach_err = paddle_pos - ball_pos
        palm_pos = self.data.site_xpos[:, self.grasp_sid]
        palm_err = palm_pos - paddle_pos

        noised_palm_pos = palm_pos + torch.randn_like(palm_pos) * self.cfg.pos_jitter_range
        noised_reach_err = noised_paddle_pos - ball_pos
        noised_palm_err = noised_palm_pos - noised_paddle_pos

        target_pos = self.target_pos
        target_vel = self.target_vel
        target_spin = self.target_spin
        target_paddle_angvel = self.target_paddle_angvel
        target_time = self.target_time.unsqueeze(-1)

        act = self.data.act.clone()
        noised_act = act + torch.randn_like(act) * self.cfg.act_jitter_range

        actuator_length = self.data.actuator_length
        actuator_velocity = self.data.actuator_velocity

        critic_obs_dict = {
            "time": sim_time.unsqueeze(-1),
            "pelvis_pos": pelvis_pos,
            "body_qpos": body_qpos,
            "body_qvel": body_qvel,
            "ball_pos": ball_pos,
            "ball_vel": ball_vel,
            "ball_spin": ball_spin,
            "paddle_pos": paddle_pos,
            "paddle_vel": paddle_vel,
            "paddle_angvel": paddle_angvel,
            "paddle_ori": paddle_ori,
            "reach_err": reach_err,
            "palm_err": palm_err,
            "touching_info": self.touching_info,
            "act": act,
            "actuator_length": actuator_length,
            "actuator_velocity": actuator_velocity,
            "target_pos": target_pos,
            "target_vel": target_vel,
            "target_spin": target_spin,
            "target_paddle_angvel": target_paddle_angvel,
            "target_time": target_time,
            "paddle_mass": self.model.body_mass[:, self.paddle_bid].unsqueeze(-1),
            "ball_friction": self.model.geom_friction[:, self.ball_gid],
            "ball_physics_params": torch.stack(
                (self.drag_scale, self.magnus_scale, self.table_friction_physics, self.racket_grip_scale),
                dim=-1,
            ),
        }

        if self.cfg.enable_domain_randomization:
            obs_dict = {
                "time": sim_time.unsqueeze(-1),
                "pelvis_pos": noised_pelvis_pos,
                "body_qpos": noised_body_qpos,
                "body_qvel": noised_body_qvel,
                "ball_pos": ball_pos,
                "ball_vel": ball_vel,
                "ball_spin": noised_ball_spin,
                "paddle_pos": noised_paddle_pos,
                "paddle_vel": noised_paddle_vel,
                "paddle_angvel": noised_paddle_angvel,
                "paddle_ori": paddle_ori,
                "reach_err": noised_reach_err,
                "palm_err": noised_palm_err,
                "touching_info": self.touching_info,
                "act": noised_act,
                "target_pos": target_pos,
                "target_vel": target_vel,
                "target_spin": target_spin,
                "target_paddle_angvel": target_paddle_angvel,
                "target_time": target_time,
            }
        else:
            obs_dict = critic_obs_dict

        obs_list = list([obs_dict[k].clone() for k in self.cfg.obs_keys])
        critic_obs_list = list([critic_obs_dict[k].clone() for k in self.cfg.critic_obs_keys])

        self.current_obs = torch.cat(obs_list, dim=-1).nan_to_num(0)
        self.extras = {
            "observations": {"critic": torch.cat(critic_obs_list, dim=-1).nan_to_num(0)},
            "obs_dict": obs_dict,
            "log": {},
            "time_outs": torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
        }

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        # obs = torch.cat([self.last_obs, self.current_obs], dim=-1)

        obs = self.current_obs
        return obs, self.extras

    def _rand_ball_pos_and_vel(self, n_reset_envs: int):
        ball_qpos = (
            torch.rand((n_reset_envs, 3)).to(self.device) * (self.ball_xyz_high - self.ball_xyz_low) + self.ball_xyz_low
        )
        table_upper = torch.tensor([1.35, 0.50, self.cfg.table_contact_height]).to(self.device)
        # Aim the serve bounce near the far edge of the robot half.  With the
        # calibrated drag/contact model, older full-half sampling placed most
        # balls below the 1.1 m reachable hit zone by x=1.8 m.
        table_lower = torch.tensor([1.15, -0.40, self.cfg.table_contact_height]).to(self.device)
        gravity = 9.81
        v_z = torch.rand((n_reset_envs,)).to(self.device) * 0.2 + 0.6

        a = -0.5 * gravity
        b = v_z
        c = ball_qpos[:, 2] - table_upper[2]

        discriminant = b**2 - 4 * a * c
        t = (-b - discriminant**0.5) / (2 * a)

        if (discriminant < 0).any():
            raise ValueError(f"No real t: z0={ball_qpos[:, 2]}, z_target={table_upper[2]}, v_z_init={v_z}")

        v_upper = torch.stack(
            [(table_upper[0] - ball_qpos[:, 0]) / t, (table_upper[1] - ball_qpos[:, 1]) / t, v_z], dim=-1
        )
        v_lower = torch.stack(
            [(table_lower[0] - ball_qpos[:, 0]) / t, (table_lower[1] - ball_qpos[:, 1]) / t, v_z], dim=-1
        )
        ball_qvel = torch.rand((n_reset_envs, 3)).to(self.device) * (v_upper - v_lower) + v_lower

        return ball_qpos, ball_qvel

    def _rand_ball_vel(self, n_reset_envs: int, ball_qpos: torch.Tensor):

        table_upper = torch.tensor([1.35, 0.50, self.cfg.table_contact_height]).to(self.device)
        table_lower = torch.tensor([1.15, -0.40, self.cfg.table_contact_height]).to(self.device)
        gravity = 9.81
        v_z = torch.rand((n_reset_envs,)).to(self.device) * 0.2 + 0.4

        a = -0.5 * gravity
        b = v_z
        c = ball_qpos[:, 2] - table_upper[2]

        discriminant = b**2 - 4 * a * c
        t = (-b - discriminant**0.5) / (2 * a)

        if (discriminant < 0).any():
            print(f"ball_qpos: {ball_qpos}")
            raise ValueError(f"No real t: z0={ball_qpos[:, 2]}, z_target={table_upper[2]}, v_z_init={v_z}")

        v_upper = torch.stack(
            [(table_upper[0] - ball_qpos[:, 0]) / t, (table_upper[1] - ball_qpos[:, 1]) / t, v_z], dim=-1
        )
        v_lower = torch.stack(
            [(table_lower[0] - ball_qpos[:, 0]) / t, (table_lower[1] - ball_qpos[:, 1]) / t, v_z], dim=-1
        )
        ball_qvel = torch.rand((n_reset_envs, 3)).to(self.device) * (v_upper - v_lower) + v_lower

        return ball_qvel



    def _get_termination_train(self) -> torch.Tensor:
        """Termination condition used for training, where the episode ends after 10s"""
        # the paddle did not touch the ball after hit time + tolerance
        ball_miss = (self.data.time > self.target_time + self.target_time_tolerance) & (self.touching_state == 0)

        # over max time limit
        max_time = self.data.time > 5

        # the position of ball is out of range
        ball_pos = self.data.site_xpos[:, self.ball_sid]
        ball_pos_in_range = (ball_pos >= self.ball_limited_low) & (ball_pos <= self.ball_limited_high)
        ball_pos_out_of_range = (~ball_pos_in_range).any(dim=-1)

        # the ball touched the net or left the play area after leaving the paddle

        # leave the paddle
        leave_paddle = (self.touching_state == 2) & (self.after_leaving_paddle >= self.leaving_paddle_tolerance)

        return max_time | ball_miss | ball_pos_out_of_range | leave_paddle

    def _get_termination_test(self) -> torch.Tensor:
        """Termination condition used for testing, where the episode ends after hitting the table"""
        # the paddle did not touch the ball after hit time + tolerance
        ball_miss = (self.data.time > self.target_time + self.target_time_tolerance) & (self.touching_state == 0)

        # over max time limit
        max_time = self.data.time > 5

        # the position of ball is out of range
        ball_pos = self.data.site_xpos[:, self.ball_sid]
        ball_pos_in_range = (ball_pos >= self.ball_limited_low) & (ball_pos <= self.ball_limited_high)
        ball_pos_out_of_range = (~ball_pos_in_range).any(dim=-1)

        # the ball touched the net or left the play area after leaving the paddle

        return max_time | ball_miss | ball_pos_out_of_range

    def _paddle_contact_velocity(self) -> torch.Tensor:
        """Return the velocity of the planned contact point on the paddle.

        ``v_body + omega x r`` is the only part of the racket twist the ball
        responds to, so it is what the command reward scores.  The lever arm is
        stored in the paddle body frame and rotated here, because the paddle has
        moved since the plan was made.
        """

        _, _, paddle_angvel, _, _ = self._paddle_kinematics()
        paddle_quat = self.data.qpos[:, self.paddle_posadr + 3 : self.paddle_posadr + 7]
        contact_offset = math.quat_apply(paddle_quat, self.target_contact_offset_local)
        body_velocity = self.data.qvel[:, self.paddle_dofadr : self.paddle_dofadr + 3]
        return body_velocity + torch.cross(paddle_angvel, contact_offset, dim=-1)

    def _cal_reward(self) -> tuple[torch.Tensor, dict]:
        """Calculate the reward"""

        # dense rewards
        rel_pos, rel_quat = self._cal_paddle_hand_rel_pose()
        rel_pos_err = torch.norm(rel_pos - self.init_rel_pos, dim=-1)
        rel_quat_err = 2 * torch.arccos(torch.clamp(torch.abs(torch.sum(rel_quat * self.init_rel_quat, dim=-1)), 0, 1))

        fin_open = self._get_fin_open()
        
        # if leave the paddle, paddle_ori_err and paddle_pos_err should be 0
        paddle_ori_err = self._get_paddle_ori_err()
        paddle_pos_err = self._get_paddle_pos_err()
        paddle_ori_err[self.touching_state >= 1] = 0
        paddle_pos_err[self.touching_state >= 1] = 0

        # sparse rewards
        self.hit_with_paddle_count += (self.touching_state >= 1).int()
        fall_plane_dist = torch.zeros(self.num_envs, device=self.device).float()
        fall_hit_plane = torch.zeros(self.num_envs, device=self.device).float()
        fall_opponent = torch.zeros(self.num_envs, device=self.device).float()
        paddle_contact_vel_err = torch.zeros(self.num_envs, device=self.device).float()
        spin_target_reward = torch.zeros(self.num_envs, device=self.device).float()
        spin_error = torch.zeros(self.num_envs, device=self.device).float()
        landing_target_reward = torch.zeros(self.num_envs, device=self.device).float()
        landing_error = torch.zeros(self.num_envs, device=self.device).float()
        net_penalty = torch.zeros(self.num_envs, device=self.device).float()

        reward_vel_mask = (self.data.time - self.target_time).abs() < 0.005
        if reward_vel_mask.any():
            # Only the contact-point velocity `v_body + omega x r` acts on the
            # ball; how the planner splits it between arm translation and wrist
            # rotation is one arbitrary point on a continuum of equivalent
            # solutions.  Scoring the split would constrain a null mode, and
            # scoring `v_body` alone silently ignores the quarter of the contact
            # velocity that rotation contributes.
            contact_velocity = self._paddle_contact_velocity()
            paddle_contact_vel_err[reward_vel_mask] = torch.exp(
                -0.5
                * torch.norm(
                    contact_velocity[reward_vel_mask]
                    - self.target_contact_vel[reward_vel_mask],
                    dim=-1,
                )
            )

        # leaving the paddle and reach tolerance, calculate fall opponent reward
        if ((self.touching_state == 2) & (self.after_leaving_paddle == self.leaving_paddle_tolerance)).any():
            ball_pos, ball_vel, ball_spin = (
                value.clone() for value in self._ball_kinematics()
            )

            # index mask of env ids to use planner to calculate fall opponent reward
            touching_mask = (self.touching_state == 2) & (self.after_leaving_paddle == self.leaving_paddle_tolerance)
            if self.cfg.spin_physics_enabled:
                t_land, land_pos, land_vel, land_spin, valid_land = predict_land(
                    ball_pos[touching_mask],
                    ball_vel[touching_mask],
                    ball_spin[touching_mask],
                    event_root_iterations=self.cfg.spin_planner_root_iterations,
                    event_integration_substeps=self.cfg.spin_planner_integration_substeps,
                    physics=self.ball_physics,
                )
                _, net_pos, _, _, crossed_net = propagate_to_x(
                    ball_pos[touching_mask],
                    ball_vel[touching_mask],
                    ball_spin[touching_mask],
                    0.0,
                    max_time=1.0,
                    params=self.ball_physics,
                    root_iterations=self.cfg.spin_planner_root_iterations,
                    integration_substeps=self.cfg.spin_planner_integration_substeps,
                )
                fail = (
                    ~crossed_net
                    | (net_pos[:, 1] > 0.80)
                    | (net_pos[:, 1] < -0.72)
                    | (net_pos[:, 2] < MIN_RETURN_NET_HEIGHT)
                )
            else:
                t_land, land_pos, land_vel = compute_land(ball_pos[touching_mask], ball_vel[touching_mask])
                land_spin = ball_spin[touching_mask]
                valid_land = torch.ones_like(t_land, dtype=torch.bool)
                fail = compute_land_net(
                    ball_pos[touching_mask],
                    ball_vel[touching_mask],
                    net_h=MIN_RETURN_NET_HEIGHT,
                )

            on_opponent_half = (
                (land_pos[:, 1] > -0.68) & (land_pos[:, 1] < 0.76) & (land_pos[:, 0] > -1.34) & (land_pos[:, 0] < -0.05)
            )
            success = valid_land & on_opponent_half & ~fail
            net_penalty[touching_mask] = fail.float()

            touching_idx = torch.nonzero(touching_mask, as_tuple=True)[0]
            fall_opponent[touching_mask] = success.float()

            target_land = self.target_landing_pos[touching_mask]
            target_error = torch.linalg.vector_norm(
                land_pos[..., :2] - target_land[..., :2], dim=-1
            ).nan_to_num(1.0e3)
            landing_error[touching_mask] = target_error
            landing_target_reward[touching_mask] = torch.exp(-0.5 * (target_error / 0.20) ** 2) * success

            hit_plane_pos = torch.zeros_like(land_pos)
            hit_plane_valid = torch.zeros_like(success)
            if success.any():
                successful = torch.nonzero(success, as_tuple=True)[0]
                if self.cfg.spin_physics_enabled:
                    bounce_vel, bounce_spin, _ = table_impact(
                        land_vel[successful],
                        land_spin[successful],
                        params=self.ball_physics,
                    )
                    _, successful_hit_pos, _, _, successful_hit_valid = propagate_to_x(
                        land_pos[successful],
                        bounce_vel,
                        bounce_spin,
                        -1.8,
                        max_time=1.0,
                        params=self.ball_physics,
                        root_iterations=self.cfg.spin_planner_root_iterations,
                        integration_substeps=self.cfg.spin_planner_integration_substeps,
                    )
                else:
                    bounce_vel = land_vel[successful].clone()
                    bounce_vel[:, 2] = -bounce_vel[:, 2]
                    successful_hit_pos, _, _ = compute_hit_pos(
                        t_land[successful],
                        land_pos[successful],
                        bounce_vel,
                        hit_plane_x=-1.8,
                    )
                    successful_hit_valid = torch.isfinite(successful_hit_pos).all(dim=-1)
                hit_plane_pos[successful] = successful_hit_pos.nan_to_num(0.0)
                hit_plane_valid[successful] = successful_hit_valid

            on_hit_plane = (
                success
                & hit_plane_valid
                & (hit_plane_pos[:, 1] > -0.7)
                & (hit_plane_pos[:, 1] < 0.7)
                & (hit_plane_pos[:, 2] > 1.1)
                & (hit_plane_pos[:, 2] < 1.35)
            )
            fall_hit_plane[touching_idx] = on_hit_plane.float()
            fall_plane_dist[touching_idx] = (
                torch.exp(-0.5 * torch.norm(hit_plane_pos[:, 1:] - self.plane_center[1:], dim=-1))
                * success
            )

            if self.cfg.spin_physics_enabled and self.cfg.spin_reward_enabled:
                measured_vel = torch.where(
                    self.has_post_hit_state[touching_mask].unsqueeze(-1),
                    self.post_hit_ball_vel[touching_mask],
                    ball_vel[touching_mask],
                )
                measured_spin = torch.where(
                    self.has_post_hit_state[touching_mask].unsqueeze(-1),
                    self.post_hit_ball_spin[touching_mask],
                    ball_spin[touching_mask],
                )
                achieved_spin = world_spin_to_trajectory(measured_spin, measured_vel)
                command_error = achieved_spin - self.target_spin[touching_mask]
                normalized_spin_error = command_error / self.spin_reward_sigma
                spin_target_reward[touching_mask] = (
                    torch.exp(
                        -0.5
                        * torch.sum(
                            self.spin_reward_axis_weights * normalized_spin_error**2,
                            dim=-1,
                        )
                    )
                    * success
                )
                spin_error[touching_mask] = torch.sqrt(
                    torch.sum(self.spin_reward_axis_weights * command_error**2, dim=-1)
                )


        act_reg = torch.norm(self.data.act, dim=-1) / self.model.na

        reward_dict = {
            "rel_pos_err": torch.exp(-20.0 * rel_pos_err),
            "rel_quat_err": torch.exp(-2.0 * rel_quat_err),
            "fin_open": torch.exp(-5.0 * fin_open),
            "paddle_pos_err": torch.exp(-8.0 * paddle_pos_err),
            "paddle_ori_err": torch.exp(-2.0 * paddle_ori_err),
            "hit_with_paddle": (self.hit_with_paddle_count == 1).float(),
            "spin_target": spin_target_reward,
            "spin_error": spin_error,
            "fall_hit_plane": fall_hit_plane,
            "paddle_contact_vel_err": paddle_contact_vel_err,
            "fall_opponent": fall_opponent,
            "fall_plane_dist": fall_plane_dist,
            "landing_target": landing_target_reward,
            "landing_error": landing_error,
            "net_penalty": net_penalty,
            "act_reg": act_reg,
        }

        shaping_scale = self._get_shaping_scale()
        dense_command_keys = {
            "paddle_pos_err",
            "paddle_ori_err",
            "paddle_contact_vel_err",
        }
        reward = torch.sum(
            torch.stack(
                [
                    reward_dict[key]
                    * self.cfg.weighted_reward_keys[key]
                    * (shaping_scale if key in dense_command_keys else 1.0)
                    for key in self.cfg.weighted_reward_keys
                ],
                dim=-1,
            ),
            dim=-1,
        )

        return reward.nan_to_num(0), reward_dict

    def _cal_paddle_hand_rel_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the relative position and orientation between paddle and hand"""
        paddle_pos = self.data.site_xpos[:, self.paddle_sid]
        paddle_ori = self.data.site_xmat[:, self.paddle_sid].reshape(-1, 3, 3)
        hand_pos = self.data.site_xpos[:, self.grasp_sid]
        hand_ori = self.data.site_xmat[:, self.grasp_sid].reshape(-1, 3, 3)

        paddle_hand_rel_ori = torch.bmm(hand_ori.transpose(1, 2), paddle_ori)
        paddle_hand_rel_quat = math.quat_from_matrix(paddle_hand_rel_ori)
        paddle_hand_rel_pos = torch.bmm(hand_ori.transpose(1, 2), (paddle_pos - hand_pos).unsqueeze(-1)).squeeze(-1)

        return paddle_hand_rel_pos, paddle_hand_rel_quat

    def _get_fin_open(self) -> torch.Tensor:
        palm_pos = self.data.site_xpos[:, self.palm_sid]
        fin0_err = torch.norm(self.data.site_xpos[:, self.fin0_sid] - palm_pos, dim=-1)
        fin1_err = torch.norm(self.data.site_xpos[:, self.fin1_sid] - palm_pos, dim=-1)
        fin2_err = torch.norm(self.data.site_xpos[:, self.fin2_sid] - palm_pos, dim=-1)
        fin3_err = torch.norm(self.data.site_xpos[:, self.fin3_sid] - palm_pos, dim=-1)
        fin4_err = torch.norm(self.data.site_xpos[:, self.fin4_sid] - palm_pos, dim=-1)
        fin_open = fin0_err + fin1_err + fin2_err + fin3_err + fin4_err

        return fin_open

    def _get_paddle_ori_err(self) -> torch.Tensor:
        if not self.cfg.spin_physics_enabled or not self.cfg.spin_planner_enabled:
            paddle_face_dir = torch.bmm(
                self.data.site_xmat[:, self.paddle_sid].reshape(-1, 3, 3),
                self.init_paddle_face_dir.unsqueeze(-1),
            ).squeeze(-1)
            return torch.arccos(
                torch.clamp(
                    torch.abs(
                        torch.sum(paddle_face_dir * self.target_vel, dim=-1)
                        / (torch.norm(self.target_vel, dim=-1) + 1.0e-6)
                    ),
                    0.0,
                    1.0,
                )
            )

        paddle_ori = self.data.qpos[
            :, self.paddle_posadr + 3 : self.paddle_posadr + 7
        ]
        local_face = self.paddle_face_dir_local.expand(self.num_envs, -1)
        paddle_normal = math.quat_apply(paddle_ori, local_face)
        target_normal = math.quat_apply(self.target_ori, local_face)
        normal_dot = torch.sum(paddle_normal * target_normal, dim=-1)
        # Roll about the face normal and either physical face are equivalent in
        # the isotropic contact model, so do not constrain those null modes.
        return torch.arccos(torch.clamp(torch.abs(normal_dot), 0.0, 1.0))

    def _get_paddle_pos_err(self) -> torch.Tensor:
        paddle_pos = self.data.qpos[:, self.paddle_posadr : self.paddle_posadr + 3]
        if self.cfg.spin_physics_enabled and self.cfg.spin_planner_enabled:
            return torch.norm(paddle_pos - self.target_pos, dim=-1)

        target_vel_dir = self.target_vel / (
            torch.norm(self.target_vel, dim=-1, keepdim=True) + 1.0e-6
        )
        real_target_pos = torch.zeros_like(self.target_pos)
        vel_to_opponent = target_vel_dir[:, 0] < 0
        real_target_pos[vel_to_opponent] = (
            self.target_pos[vel_to_opponent] - target_vel_dir[vel_to_opponent] * 0.04
        )
        real_target_pos[~vel_to_opponent] = (
            self.target_pos[~vel_to_opponent] + target_vel_dir[~vel_to_opponent] * 0.04
        )
        return torch.norm(paddle_pos - real_target_pos, dim=-1)

    
    def _check_ball_cross_net(self, ball_qpos: torch.Tensor, ball_qvel: torch.Tensor) -> torch.Tensor:
        """Check if the ball crosses the net"""
        t_net = (0.0 - ball_qpos[:, 0]) / ball_qvel[:, 0]
        h_net = ball_qpos[:, 2] + ball_qvel[:, 2] * t_net + 0.5 * -9.81 * t_net**2
        cross_net_flag = h_net > MIN_SERVE_NET_HEIGHT
        return cross_net_flag


    def _get_spin_curriculum_scale(self) -> float:
        if not self.cfg.spin_curriculum_enabled or self.cfg.eval_env:
            return 1.0
        current_iteration = float(getattr(self, "curr_iter", 0))
        total_iterations = max(float(getattr(self, "total_iter", 1)), 1.0)
        full_fraction = max(float(self.cfg.spin_curriculum_full_fraction), 1.0e-6)
        progress = min(current_iteration / total_iterations / full_fraction, 1.0)
        initial_scale = float(self.cfg.spin_curriculum_initial_scale)
        return initial_scale + (1.0 - initial_scale) * progress

    def _get_shaping_scale(self) -> float:
        """Anneal planner-imitation rewards while preserving task rewards."""

        if not self.cfg.spin_physics_enabled or self.cfg.eval_env:
            return 1.0
        current_iteration = float(getattr(self, "curr_iter", 0))
        total_iterations = max(float(getattr(self, "total_iter", 1)), 1.0)
        full_fraction = max(float(self.cfg.shaping_full_fraction), 1.0e-6)
        progress = min(current_iteration / total_iterations / full_fraction, 1.0)
        final_scale = float(self.cfg.shaping_final_scale)
        return 1.0 + (final_scale - 1.0) * progress


    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset environment, resample the domain randomization parameters"""
        
        n_reset_envs = env_ids.shape[0]

        # clear episode info
        self.episode_length_buf[env_ids] = 0
        self.touching_state[env_ids] = 0
        self.after_leaving_paddle[env_ids] = 0
        self.landing_state[env_ids] = 0
        self.hit_with_paddle_count[env_ids] = 0
        self.touching_info[env_ids] = 0
        self.after_leaving_own[env_ids] = 0
        self._substep_contact_info[env_ids] = False
        self._contact_latched[env_ids] = False
        self._contact_separation_steps[env_ids] = 0
        self.post_hit_ball_vel[env_ids] = 0.0
        self.post_hit_ball_spin[env_ids] = 0.0
        self.has_post_hit_state[env_ids] = False

        if self.cfg.enable_domain_randomization:
            def sample_scalar(bounds):
                return torch.rand(n_reset_envs, device=self.device) * (bounds[1] - bounds[0]) + bounds[0]

            self.drag_scale[env_ids] = sample_scalar(self.cfg.drag_scale_range)
            self.magnus_scale[env_ids] = sample_scalar(self.cfg.magnus_scale_range)
            self.table_friction_physics[env_ids] = sample_scalar(self.cfg.table_friction_physics_range)
            self.racket_grip_scale[env_ids] = sample_scalar(self.cfg.racket_grip_scale_range)
        else:
            self.drag_scale[env_ids] = 1.0
            self.magnus_scale[env_ids] = 1.0
            self.table_friction_physics[env_ids] = DEFAULT_BALL_PHYSICS.table_friction
            self.racket_grip_scale[env_ids] = 1.0

        spin_curriculum_scale = self._get_spin_curriculum_scale()
        if self.cfg.spin_physics_enabled and self.cfg.spin_command_mode == "uniform":
            self.target_spin[env_ids] = (
                torch.rand((n_reset_envs, 3), device=self.device)
                * (self.spin_target_high - self.spin_target_low)
                + self.spin_target_low
            ) * spin_curriculum_scale
        elif self.cfg.spin_physics_enabled and self.cfg.spin_command_mode == "fixed":
            self.target_spin[env_ids] = (
                torch.tensor(self.cfg.spin_target, device=self.device) * spin_curriculum_scale
            )
        elif self.cfg.spin_command_mode not in ("fixed", "uniform"):
            raise ValueError(f"Unknown spin_command_mode: {self.cfg.spin_command_mode}")
        else:
            self.target_spin[env_ids] = 0.0

        self.target_landing_pos[env_ids] = (
            torch.rand((n_reset_envs, 3), device=self.device)
            * (self.opponent_table_upper - self.opponent_table_lower)
            + self.opponent_table_lower
        )

        # domain randomization for mjmodel
        # self.model.body_mass[env_ids, self.paddle_bid] = (
        #     torch.rand((n_reset_envs,)).to(self.device) * (self.paddle_mass_high - self.paddle_mass_low)
        #     + self.paddle_mass_low
        # )
        # self.model.geom_friction[env_ids, self.ball_gid] = (
        #     torch.rand((n_reset_envs, 3)).to(self.device) * (self.ball_friction_high - self.ball_friction_low)
        #     + self.ball_friction_low
        # )
        # self.model.body_mass[env_ids, self.paddle_bid] = 0.1318480843660727
        # self.model.geom_friction[env_ids, self.ball_gid] = torch.tensor([9.5396e-01, 4.0819e-03, 1.0331e-05]).to(self.device)

        # randomization on ball position, calculate every reset
        init_ball_qpos = torch.zeros(n_reset_envs, 3, device=self.device)
        init_ball_qvel = torch.zeros(n_reset_envs, 3, device=self.device)
        init_ball_spin = torch.zeros(n_reset_envs, 3, device=self.device)

        # Resample until the spin-aware trajectory clears the net and reaches the hit zone.
        cross_net_flag = torch.zeros(n_reset_envs, device=self.device, dtype=torch.bool)
        sampling_attempt = 0
        while not cross_net_flag.all().item():
            n_envs_remain = (~cross_net_flag).sum().item()
            # Oversampling prevents the last hard-to-fill environment from
            # causing dozens of tiny GPU launches and host synchronizations.
            n_candidates = max(4 * n_envs_remain, 64)
            init_ball_qpos_remain, init_ball_qvel_remain = self._rand_ball_pos_and_vel(
                n_candidates
            )
            if self.cfg.spin_physics_enabled:
                if self.cfg.incoming_spin_enabled:
                    incoming_spin_command = (
                        torch.rand((n_candidates, 3), device=self.device)
                        * (self.incoming_spin_high - self.incoming_spin_low)
                        + self.incoming_spin_low
                    ) * spin_curriculum_scale
                else:
                    incoming_spin_command = torch.zeros(
                        n_candidates, 3, device=self.device
                    )
                init_ball_spin_remain = trajectory_spin_to_world(incoming_spin_command, init_ball_qvel_remain)
                _, net_pos, _, _, crossed_net = propagate_to_x(
                    init_ball_qpos_remain,
                    init_ball_qvel_remain,
                    init_ball_spin_remain,
                    0.0,
                    max_time=1.0,
                    params=self.ball_physics,
                    root_iterations=self.cfg.spin_planner_root_iterations,
                    integration_substeps=self.cfg.spin_planner_integration_substeps,
                )
                cross_net_flag_remain = crossed_net & (net_pos[:, 2] > MIN_SERVE_NET_HEIGHT)
                incoming = predict_incoming_hit(
                    init_ball_qpos_remain,
                    init_ball_qvel_remain,
                    init_ball_spin_remain,
                    event_root_iterations=self.cfg.spin_planner_root_iterations,
                    event_integration_substeps=self.cfg.spin_planner_integration_substeps,
                    physics=self.ball_physics,
                )
                hit_pos = incoming.position
                cross_net_flag_remain &= incoming.valid
            else:
                init_ball_spin_remain = torch.zeros_like(init_ball_qvel_remain)
                cross_net_flag_remain = self._check_ball_cross_net(init_ball_qpos_remain, init_ball_qvel_remain)
                t_land, land_pos, land_vel = compute_land(init_ball_qpos_remain, init_ball_qvel_remain)
                bounce_vel = land_vel.clone()
                bounce_vel[:, 2] = -bounce_vel[:, 2]
                hit_pos, _, _ = compute_hit_pos(t_land, land_pos, bounce_vel)

            target_in_range = (
                (hit_pos[:, 1] > self.hit_xyz_low[1])
                & (hit_pos[:, 1] < self.hit_xyz_high[1])
                & (hit_pos[:, 2] > self.hit_xyz_low[2])
                & (hit_pos[:, 2] < self.hit_xyz_high[2])
            )
            
            cross_net_flag_remain = cross_net_flag_remain & target_in_range


            n_success = min(cross_net_flag_remain.sum().item(), n_envs_remain)

            fail_indices = torch.where(~cross_net_flag)[0]
            successful_candidates = torch.where(cross_net_flag_remain)[0][:n_success]
            init_ball_qpos[fail_indices[:n_success]] = init_ball_qpos_remain[
                successful_candidates
            ]
            init_ball_qvel[fail_indices[:n_success]] = init_ball_qvel_remain[
                successful_candidates
            ]
            init_ball_spin[fail_indices[:n_success]] = init_ball_spin_remain[
                successful_candidates
            ]
            cross_net_flag[fail_indices[:n_success]] = True
            sampling_attempt += 1
            if sampling_attempt >= 20 and not cross_net_flag.all().item():
                raise RuntimeError("Could not sample valid incoming spin trajectories")

        # local_env_ids = torch.arange(n_reset_envs, device=self.device)
        # init_ball_qpos[local_env_ids] = torch.tensor([-0.9634, -0.2302, 1.4041]).to(self.device)
        # init_ball_qvel[local_env_ids] = torch.tensor([6.2354, 2.3637, -0.0967]).to(self.device)
        # self.model.body_pos[env_ids, self.ball_bid] = init_ball_qpos

        self.data.time[env_ids] = 0.0
        self.data.qpos[env_ids] = self.init_qpos
        self.data.qpos[env_ids, self.ball_posadr : self.ball_posadr + 3] = init_ball_qpos
        self.data.qvel[env_ids] = 0.0
        self.data.qvel[env_ids, self.ball_dofadr : self.ball_dofadr + 3] = init_ball_qvel
        ball_quat = self.data.qpos[env_ids, self.ball_posadr + 3 : self.ball_posadr + 7]
        self.data.qvel[env_ids, self.ball_dofadr + 3 : self.ball_dofadr + 6] = (
            math.quat_apply_inverse(ball_quat, init_ball_spin)
        )
        self.data.qfrc_applied[env_ids] = 0
        self.data.xfrc_applied[env_ids] = 0
        self.data.ctrl[env_ids] = 0.0
        self.data.act[env_ids] = 0.0
        self.sim.forward()

        # get high command
        (
            paddle_pos,
            paddle_vel,
            paddle_angvel,
            paddle_ori,
            hit_time,
            hit_pos,
            target_spin_world,
            contact_vel,
            contact_offset_local,
            plan_valid,
        ) = self.get_high_command(
            init_ball_qpos,
            init_ball_qvel,
            init_ball_spin,
            spin_command=self.target_spin[env_ids],
            target_landing=self.target_landing_pos[env_ids],
            return_valid=True,
        )

        if self.cfg.spin_physics_enabled and self.cfg.spin_planner_enabled:
            planning_attempt = 0
            while not plan_valid.all().item() and planning_attempt < 8:
                retry = ~plan_valid
                retry_count = retry.sum().item()
                self.target_landing_pos[env_ids[retry]] = (
                    torch.rand((retry_count, 3), device=self.device)
                    * (self.opponent_table_upper - self.opponent_table_lower)
                    + self.opponent_table_lower
                )
                if self.cfg.spin_command_mode == "uniform":
                    self.target_spin[env_ids[retry]] = (
                        torch.rand((retry_count, 3), device=self.device)
                        * (self.spin_target_high - self.spin_target_low)
                        + self.spin_target_low
                    ) * spin_curriculum_scale
                retry_command = self.get_high_command(
                    init_ball_qpos[retry],
                    init_ball_qvel[retry],
                    init_ball_spin[retry],
                    spin_command=self.target_spin[env_ids[retry]],
                    target_landing=self.target_landing_pos[env_ids[retry]],
                    return_valid=True,
                )
                retry_tensors = retry_command[:-1]
                for destination, source in zip(
                    (
                        paddle_pos,
                        paddle_vel,
                        paddle_angvel,
                        paddle_ori,
                        hit_time,
                        hit_pos,
                        target_spin_world,
                        contact_vel,
                        contact_offset_local,
                    ),
                    retry_tensors,
                    strict=True,
                ):
                    destination[retry] = source
                plan_valid[retry] = retry_command[-1]
                planning_attempt += 1
            if not plan_valid.all().item():
                raise RuntimeError("Could not sample feasible spin return plans")

        self.target_pos[env_ids] = paddle_pos
        self.target_vel[env_ids] = paddle_vel
        self.target_paddle_angvel[env_ids] = paddle_angvel
        self.target_ori[env_ids] = paddle_ori
        self.target_spin_world[env_ids] = target_spin_world
        self.target_contact_vel[env_ids] = contact_vel
        self.target_contact_offset_local[env_ids] = contact_offset_local
        self.target_plan_valid[env_ids] = plan_valid
        self.hit_pos[env_ids] = hit_pos
        self.target_time[env_ids] = hit_time

    
    def get_high_command(
        self,
        init_ball_qpos,
        init_ball_qvel,
        init_ball_spin=None,
        *,
        start_time=None,
        already_bounced=False,
        spin_command=None,
        target_landing=None,
        return_valid=False,
    ):
        if self.cfg.spin_physics_enabled and self.cfg.spin_planner_enabled:
            if init_ball_spin is None:
                init_ball_spin = torch.zeros_like(init_ball_qvel)
            if spin_command is None:
                spin_command = torch.zeros_like(init_ball_qvel)
            if target_landing is None:
                target_landing = self.opponent_table_lower.expand(init_ball_qpos.shape[0], -1)
            incoming = predict_incoming_hit(
                init_ball_qpos,
                init_ball_qvel,
                init_ball_spin,
                start_time=start_time,
                already_bounced=already_bounced,
                table_height=self.cfg.table_contact_height,
                event_root_iterations=self.cfg.spin_planner_root_iterations,
                event_integration_substeps=self.cfg.spin_planner_integration_substeps,
                physics=self.ball_physics,
            )
            plan = plan_spin_return(
                incoming,
                target_landing,
                spin_command,
                paddle_face_local=self.paddle_face_dir_local.expand(init_ball_qpos.shape[0], -1),
                spin_priority=self.cfg.spin_priority,
                wrist_fraction=self.cfg.planner_wrist_fraction,
                max_paddle_speed=self.cfg.planner_max_paddle_speed,
                max_paddle_angular_speed=self.cfg.planner_max_paddle_angular_speed,
                event_root_iterations=self.cfg.spin_planner_root_iterations,
                event_integration_substeps=self.cfg.spin_planner_integration_substeps,
                physics=self.ball_physics,
            )
            target_spin_world = trajectory_spin_to_world(spin_command, plan.predicted_out_velocity)
            result = (
                plan.paddle_position,
                plan.paddle_velocity,
                plan.paddle_spin,
                plan.paddle_orientation,
                plan.hit_time,
                plan.hit_position,
                target_spin_world,
                plan.contact_velocity,
                plan.contact_offset_local,
                plan.valid,
            )
            return result if return_valid else result[:-1]

        if not already_bounced:
            t_land, land_pos, land_vel = compute_land(init_ball_qpos, init_ball_qvel)
            bounce_vel = land_vel.clone()
            bounce_vel[:, 2] = -bounce_vel[:, 2]
        else:
            land_pos = init_ball_qpos
            land_vel = init_ball_qvel
            bounce_vel = land_vel.clone()
            t_land = start_time

        hit_pos, v_in, hit_time = compute_hit_pos(t_land, land_pos, bounce_vel)
        paddle_vel = compute_paddle_vel(hit_pos, v_in, self.opponent_table_upper, self.opponent_table_lower, self.device)

        # if paddle is moving backward, reverse the velocity
        paddle_vel_for_ori = paddle_vel.clone()
        if (paddle_vel_for_ori[:, 0] > 0).any():
            back_mask = paddle_vel_for_ori[:, 0] > 0
            paddle_vel_for_ori[back_mask] = -paddle_vel_for_ori[back_mask]

        paddle_face_dir_local = self.paddle_face_dir_local.expand(paddle_vel_for_ori.shape[0], -1)
        paddle_ori = vec_to_quat(paddle_face_dir_local, paddle_vel_for_ori)

        paddle_pos = compute_paddle_pos(paddle_ori, hit_pos)

        result = (
            paddle_pos,
            paddle_vel,
            torch.zeros_like(paddle_vel),
            paddle_ori,
            hit_time,
            hit_pos,
            torch.zeros_like(paddle_vel),
            # No wrist twist is planned here, so the contact point moves with the
            # body and the lever arm drops out: the reward degrades to the
            # historical body-velocity comparison.
            paddle_vel,
            torch.zeros_like(paddle_vel),
            torch.ones(init_ball_qpos.shape[0], device=self.device, dtype=torch.bool),
        )
        return result if return_valid else result[:-1]


    def reset(self) -> tuple[torch.Tensor, dict]:
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self.init_rel_pos, self.init_rel_quat = self._cal_paddle_hand_rel_pose()
        self.ref_face_dir = torch.tensor([-1, 0, 0]).float().to(self.device)
        self.init_paddle_face_dir = torch.bmm(
            self.data.site_xmat[:, self.paddle_sid].reshape(-1, 3, 3).transpose(1, 2),
            self.ref_face_dir.unsqueeze(0).unsqueeze(-1).repeat_interleave(self.num_envs, dim=0),
        ).squeeze(-1)
        self._update_current_obs()
        self.last_obs = self.current_obs.clone()
        return self.get_observations()

    def joint_pd_to_muscle_act(self, actions: torch.Tensor) -> torch.Tensor:
        # TODO: first 2 dims control pelvis, remaining 273 dims control muscles
        # add perturbation to actions
        if self.cfg.enable_action_randomization:
            actions_perturbation = torch.randn_like(actions) * self.cfg.action_range
            actions += actions_perturbation

        actions = torch.clamp(actions, -1, 1)
        pelvis_actions = actions[:, :2].clone()
        actions = (actions + 1) / 2 * (self.action_high - self.action_low) + self.action_low
        target_length = get_target_actuator_length(
            self.model, self.fk_data, actions, self.equalities, self.action_to_qpos, self.sim.fk_forward
        )

        self.target_length = target_length.clone()

        kp_scale = self.cfg.kp_scale
        kd_scale = self.cfg.kd_scale

        activations, bias, gain, clipped_force = target_length_to_activations(
            self.model, self.data, target_length, kp_scale, kd_scale
        )

        self.bias = bias.clone()
        self.gain = gain.clone()
        self.clipped_force = clipped_force.clone()

        if self.cfg.enable_action_randomization:
            activation_perturbation = torch.randn_like(activations) * self.cfg.act_range
            activations += activation_perturbation

        activations = torch.clamp(
            activations,
            1.0 / (1.0 + torch.exp(torch.tensor(7.5, device=activations.device))),
            1.0 / (1.0 + torch.exp(torch.tensor(-2.5, device=activations.device))),
        )

        # [-1, 1] -> [-1, 0.05] (not joint range [-1, -0.05])
        pelvis_actions = (pelvis_actions + 1) / 2 * (self.non_muscle_high - self.non_muscle_low) + self.non_muscle_low

        return torch.cat([activations, pelvis_actions], dim=-1)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        actions = actions.clone()

        if self.cfg.action_type == "joint_pd":
            self.data.ctrl[:] = self.joint_pd_to_muscle_act(actions)
        elif self.cfg.action_type == "muscle_vae":
            # Map policy output a ∈ [-1, 1] to target muscle length:
            # target_length = (a + 1.0) * l_M^{tpose}
            # (a=0 → t-pose length, a=-1 → 0, a=1 → 2×t-pose)
            muscle_actions = actions[:, self.muscle_ind]
            pelvis_actions = actions[:, self.non_muscle_ind]

            target_length_muscles = (muscle_actions + 1.0) * self.tpose_muscle_length_muscles.unsqueeze(0)

            target_length = torch.zeros(actions.shape[0], self.data.actuator_length.shape[1], device=self.device)
            target_length[:, self.muscle_ind] = target_length_muscles
            muscle_activations = calculate_vae_muscle_act(
                model=self.model,
                data=self.data,
                target_length=target_length,
                kp=self.cfg.kp_vae,
                kd=self.cfg.kd_vae,
            )
            
            # Step 3: map pelvis actions linearly to control range
            pelvis_ctrl = (pelvis_actions + 1) / 2 * (
                self.non_muscle_high - self.non_muscle_low
            ) + self.non_muscle_low

            # Step 4: assemble full 275-dim control signal
            ctrl = torch.zeros_like(self.data.ctrl)
            ctrl[:, self.muscle_ind] = muscle_activations
            ctrl[:, self.non_muscle_ind] = pelvis_ctrl
            self.data.ctrl[:] = ctrl
        else:
            # muscle_pd or muscle_act mode
            if self.cfg.normalize_act:
                actions[:, self.muscle_ind] = 1.0 / (
                    1.0 + torch.exp(-5.0 * (actions[:, self.muscle_ind] - 0.5)).to(self.device)
                )
                actions[:, self.non_muscle_ind] = (actions[:, self.non_muscle_ind] + 1) / 2 * (
                    self.non_muscle_high - self.non_muscle_low
                ) + self.non_muscle_low
            self.data.ctrl[:] = actions

        control_contact_info = torch.zeros_like(self.touching_info)
        for _ in range(self.cfg.frame_skip):
            pre_ball_pos, pre_ball_vel, pre_ball_spin = (
                value.clone() for value in self._ball_kinematics()
            )
            pre_paddle_body_pos = self.data.qpos[
                :, self.paddle_posadr : self.paddle_posadr + 3
            ].clone()
            pre_paddle_vel = self.data.qvel[:, self.paddle_dofadr : self.paddle_dofadr + 3].clone()
            pre_paddle_quat = self.data.qpos[
                :, self.paddle_posadr + 3 : self.paddle_posadr + 7
            ].clone()
            pre_paddle_spin_local = self.data.qvel[
                :, self.paddle_dofadr + 3 : self.paddle_dofadr + 6
            ].clone()
            pre_paddle_spin = math.quat_apply(pre_paddle_quat, pre_paddle_spin_local)

            self._apply_aerodynamics()
            self.sim.step()
            substep_contact_info = self._read_touching_info()
            control_contact_info |= substep_contact_info
            rising_contact_info = self._update_contact_latches(substep_contact_info)

            if self.cfg.spin_physics_enabled and self.cfg.analytic_contact_override:
                self._apply_analytic_contacts(
                    pre_ball_pos,
                    pre_ball_vel,
                    pre_ball_spin,
                    pre_paddle_body_pos,
                    pre_paddle_vel,
                    pre_paddle_spin,
                    pre_paddle_quat,
                    rising_contact_info,
                )
            else:
                rising_paddle = rising_contact_info[:, 0]
                if rising_paddle.any():
                    _, current_ball_velocity, current_ball_spin = self._ball_kinematics()
                    self.post_hit_ball_vel[rising_paddle] = current_ball_velocity[rising_paddle]
                    self.post_hit_ball_spin[rising_paddle] = current_ball_spin[rising_paddle]
                    self.has_post_hit_state[rising_paddle] = True

        self._cal_touching_info(control_contact_info)
        self.episode_length_buf += 1


        reward, reward_dict = self._cal_reward()
        if self.cfg.eval_env:
            done = self._get_termination_test()
        else:
            done = self._get_termination_train()
        if done.any():
            self._reset_idx(done.nonzero().squeeze(-1))

        # Replan immediately after the incoming ball leaves the robot's table
        # half, before constructing the observation returned to the policy.
        landing_mask = (self.landing_state == 2) & (
            self.after_leaving_own == self.leaving_table_tolerance
        )
        if landing_mask.any():
            current_ball_pos, current_ball_vel, current_ball_spin = self._ball_kinematics()
            ball_pos_mask = current_ball_pos[landing_mask].clone()
            ball_vel_mask = current_ball_vel[landing_mask].clone()
            ball_spin_mask = current_ball_spin[landing_mask].clone()
            (
                paddle_pos,
                paddle_vel,
                paddle_angvel,
                paddle_ori,
                hit_time,
                hit_pos,
                target_spin_world,
                contact_vel,
                contact_offset_local,
                plan_valid,
            ) = self.get_high_command(
                ball_pos_mask,
                ball_vel_mask,
                ball_spin_mask,
                start_time=self.data.time[landing_mask],
                already_bounced=True,
                spin_command=self.target_spin[landing_mask],
                target_landing=self.target_landing_pos[landing_mask],
                return_valid=True,
            )
            self.target_pos[landing_mask] = paddle_pos
            self.target_vel[landing_mask] = paddle_vel
            self.target_paddle_angvel[landing_mask] = paddle_angvel
            self.target_ori[landing_mask] = paddle_ori
            self.target_spin_world[landing_mask] = target_spin_world
            self.target_contact_vel[landing_mask] = contact_vel
            self.target_contact_offset_local[landing_mask] = contact_offset_local
            self.target_plan_valid[landing_mask] = plan_valid
            self.hit_pos[landing_mask] = hit_pos
            self.target_time[landing_mask] = hit_time

        # update last obs
        self.last_obs = self.current_obs.clone()
        self._update_current_obs()
        # handle reset envs
        self.last_obs[done] = self.current_obs[done].clone()
        obs, extras = self.get_observations()

        for key in reward_dict.keys():
            extras["log"]["reward/" + key] = reward_dict[key]

        return obs, reward, done, extras

    def render_offscreen(self):
        height = 480
        width = 640
        camera = 1
        if self.renderer is None:
            self.renderer = mujoco.Renderer(
                self.sim.mj_model, height=height, width=width
            )
            self.renderer.scene.ngeom += 1
        mjw.get_data_into(self.sim.mj_data, self.sim.mj_model, self.sim.wp_data)

        self.renderer.update_scene(self.sim.mj_data, camera=camera)
        # mujoco.mjv_initGeom(
        #     self.renderer.scene.geoms[self.renderer.scene.ngeom - 1],
        #     mujoco.mjtGeom.mjGEOM_SPHERE,
        #     np.ones(3) * 0.02,
        #     self.target_pos.cpu().numpy()[0],
        #     np.eye(3).flatten(),
        #     np.array([1.0, 0.0, 0.0, 0.7]),
        # )
        return self.renderer.render()


if __name__ == "__main__":
    cfg = tabletennis_p2_cfg()
    env = TableTennisWarpEnv(cfg)
    obs, info = env.reset()


    for _ in tqdm(range(1000)):
        actions = torch.randn(env.num_envs, env.num_actions, device=env.device)
        obs, reward, done, info = env.step(actions)

    viewer = mujoco.viewer.launch_passive(env.sim.mj_model, env.sim.mj_data)
    viewer.user_scn.ngeom += 1
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.ones(3) * 0.02,
        env.target_pos.cpu().numpy()[0],
        np.eye(3).flatten(),
        np.array([1.0, 0.0, 0.0, 0.7]),
    )

    while viewer.is_running():
        actions = torch.randn(env.num_envs, env.num_actions, device=env.device)
        obs, reward, done, info = env.step(actions)
        mjw.get_data_into(env.sim.mj_data, env.sim.mj_model, env.sim.wp_data)
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom - 1],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.ones(3) * 0.02,
            env.target_pos.cpu().numpy()[0],
            np.eye(3).flatten(),
            np.array([1.0, 0.0, 0.0, 0.7]),
        )
        viewer.sync()
        time.sleep(0.01)
        if info["obs_dict"]["touching_info"][0].any():
            print(info["obs_dict"]["touching_info"][0])
            time.sleep(1.0)
