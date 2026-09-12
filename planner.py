"""Physics-based paddle planner using PyTorch for parallel computation.

The legacy gravity-only helpers remain available for reproducing old
checkpoints.  New code should use :func:`predict_incoming_hit` and
:func:`plan_spin_return`, which share the spin-aware dynamics used by the RL
environment.
"""

import math
from dataclasses import dataclass

import torch
from ball_physics import (
    DEFAULT_BALL_PHYSICS,
    BallPhysicsParams,
    propagate_to_height,
    propagate_to_x,
    racket_impact,
    table_impact,
    trajectory_spin_to_world,
)

DEFAULT_TABLE_SURFACE_HEIGHT = 0.795
DEFAULT_BALL_CONTACT_HEIGHT = DEFAULT_TABLE_SURFACE_HEIGHT + DEFAULT_BALL_PHYSICS.radius
DEFAULT_PADDLE_HALF_THICKNESS = 0.020
# Centre of the `pad` cylinder in the paddle body frame (tabletennis.xml).
PAD_CENTER_OFFSET_LOCAL = (-0.07, 0.0, 0.0)

# Top of the `coll_net` box in tabletennis.xml: centre 0.795 m, half-height
# 0.1525 m.  The two crossing thresholds below are deliberately different and
# both sit above it, so a trajectory grazing the tape is not counted as legal:
#  - the return gate is shared by the planner's validity check and the reward,
#    so a plan the planner accepts is one the reward can score;
#  - the serve gate only filters the incoming ball at reset, where a smaller
#    margin keeps the resampling acceptance rate up.
NET_TOP_HEIGHT = 0.9475
MIN_RETURN_NET_HEIGHT = 1.0  # +5.25 cm over the tape
MIN_SERVE_NET_HEIGHT = 0.98  # +3.25 cm over the tape


@dataclass
class IncomingHitPrediction:
    """Predicted incoming ball state at the robot's hit plane."""

    position: torch.Tensor
    velocity: torch.Tensor
    spin: torch.Tensor
    time: torch.Tensor
    valid: torch.Tensor


@dataclass
class SpinReturnPlan:
    """Instantaneous racket command and its predicted ball outcome."""

    paddle_position: torch.Tensor
    paddle_velocity: torch.Tensor
    paddle_spin: torch.Tensor
    paddle_orientation: torch.Tensor
    # Velocity of the contact point and the lever arm it sits on, in the paddle
    # body frame.  Only `v_body + omega x offset` acts on the ball, so the split
    # between the two above is one arbitrary point on a continuum; track this
    # pair instead of either half of it.
    contact_velocity: torch.Tensor
    contact_offset_local: torch.Tensor
    hit_position: torch.Tensor
    hit_time: torch.Tensor
    target_landing_position: torch.Tensor
    predicted_out_velocity: torch.Tensor
    predicted_out_spin: torch.Tensor
    predicted_landing_position: torch.Tensor
    valid: torch.Tensor


def _horizontal_newton_update(
    predicted_landing: torch.Tensor,
    target_landing: torch.Tensor,
    perturbed_landing: torch.Tensor,
    perturbation: float,
) -> torch.Tensor:
    """Solve a regularized 2x2 landing Jacobian for a velocity correction."""

    jacobian = (
        perturbed_landing[..., :2] - predicted_landing.unsqueeze(-2)[..., :2]
    ) / perturbation
    # Columns correspond to perturbing vx and vy respectively.
    j00, j10 = jacobian[..., 0, 0], jacobian[..., 0, 1]
    j01, j11 = jacobian[..., 1, 0], jacobian[..., 1, 1]
    error = predicted_landing[..., :2] - target_landing[..., :2]
    determinant = j00 * j11 - j01 * j10
    safe_determinant = torch.where(
        determinant.abs() > 1.0e-5,
        determinant,
        torch.where(determinant >= 0.0, 1.0e-5, -1.0e-5),
    )
    correction_x = (-j11 * error[..., 0] + j01 * error[..., 1]) / safe_determinant
    correction_y = (j10 * error[..., 0] - j00 * error[..., 1]) / safe_determinant
    correction = torch.stack((correction_x, correction_y), dim=-1).nan_to_num(0.0)
    correction_norm = torch.linalg.vector_norm(correction, dim=-1, keepdim=True)
    return correction * torch.minimum(
        torch.ones_like(correction_norm), 3.0 / correction_norm.clamp_min(1.0e-8)
    )


def predict_land(
    ball_pos: torch.Tensor,
    ball_vel: torch.Tensor,
    ball_spin: torch.Tensor,
    landing_h: float | torch.Tensor = DEFAULT_BALL_CONTACT_HEIGHT,
    *,
    max_time: float = 1.2,
    event_root_iterations: int = 4,
    event_integration_substeps: int = 3,
    physics: BallPhysicsParams = DEFAULT_BALL_PHYSICS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Spin-aware replacement for the gravity-only ``compute_land`` helper."""

    return propagate_to_height(
        ball_pos,
        ball_vel,
        ball_spin,
        landing_h,
        max_time=max_time,
        descending=True,
        params=physics,
        root_iterations=event_root_iterations,
        integration_substeps=event_integration_substeps,
    )


def predict_incoming_hit(
    ball_pos: torch.Tensor,
    ball_vel: torch.Tensor,
    ball_spin: torch.Tensor,
    *,
    start_time: torch.Tensor | None = None,
    already_bounced: bool = False,
    table_height: float = DEFAULT_BALL_CONTACT_HEIGHT,
    hit_plane_x: float = 1.8,
    max_time: float = 1.2,
    event_root_iterations: int = 4,
    event_integration_substeps: int = 3,
    physics: BallPhysicsParams = DEFAULT_BALL_PHYSICS,
) -> IncomingHitPrediction:
    """Predict the incoming state after the own-side table bounce."""

    if start_time is None:
        start_time = torch.zeros(ball_pos.shape[:-1], dtype=ball_pos.dtype, device=ball_pos.device)

    if already_bounced:
        bounce_pos, bounce_vel, bounce_spin = ball_pos, ball_vel, ball_spin
        bounce_time = torch.zeros_like(start_time)
        valid_bounce = torch.ones_like(start_time, dtype=torch.bool)
    else:
        bounce_time, bounce_pos, impact_vel, impact_spin, valid_bounce = predict_land(
            ball_pos,
            ball_vel,
            ball_spin,
            table_height,
            max_time=max_time,
            event_root_iterations=event_root_iterations,
            event_integration_substeps=event_integration_substeps,
            physics=physics,
        )
        bounce_vel, bounce_spin, _ = table_impact(impact_vel, impact_spin, params=physics)

    hit_dt, hit_pos, hit_vel, hit_spin, valid_hit = propagate_to_x(
        bounce_pos,
        bounce_vel,
        bounce_spin,
        hit_plane_x,
        max_time=max_time,
        params=physics,
        root_iterations=event_root_iterations,
        integration_substeps=event_integration_substeps,
    )
    return IncomingHitPrediction(
        position=hit_pos,
        velocity=hit_vel,
        spin=hit_spin,
        time=start_time + bounce_time + hit_dt,
        valid=valid_bounce & valid_hit & (hit_pos[..., 2] > table_height),
    )


def solve_outgoing_velocity(
    hit_pos: torch.Tensor,
    landing_pos: torch.Tensor,
    spin_command: torch.Tensor,
    *,
    launch_vertical_speed: float | torch.Tensor = 1.5,
    correction_iterations: int = 2,
    net_x: float = 0.0,
    net_clearance: float = MIN_RETURN_NET_HEIGHT,
    event_root_iterations: int = 4,
    event_integration_substeps: int = 3,
    physics: BallPhysicsParams = DEFAULT_BALL_PHYSICS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shoot a spin-aware free-flight trajectory towards a landing point.

    The solver starts from the ballistic solution and iteratively corrects the
    horizontal velocity using the shared nonlinear flight model.  It is fully
    batched and has no per-environment Python loop.
    """

    gravity = physics.gravity
    launch_vertical_speed = torch.as_tensor(
        launch_vertical_speed, dtype=hit_pos.dtype, device=hit_pos.device
    )
    launch_vertical_speed = torch.broadcast_to(launch_vertical_speed, hit_pos.shape[:-1])
    vertical_discriminant = launch_vertical_speed**2 + 2.0 * gravity * (hit_pos[..., 2] - landing_pos[..., 2])
    flight_time = (
        launch_vertical_speed + torch.sqrt(torch.clamp(vertical_discriminant, min=1.0e-8))
    ) / gravity
    flight_time = flight_time.clamp_min(0.05)
    desired_velocity = (landing_pos - hit_pos) / flight_time.unsqueeze(-1)
    desired_velocity[..., 2] = launch_vertical_speed

    valid = torch.ones(hit_pos.shape[:-1], dtype=torch.bool, device=hit_pos.device)
    predicted_landing = landing_pos.clone()
    if correction_iterations < 1:
        raise ValueError("correction_iterations must be positive")
    target_spin = trajectory_spin_to_world(spin_command, desired_velocity)
    perturbation = 0.05
    for iteration in range(correction_iterations):
        target_spin = trajectory_spin_to_world(spin_command, desired_velocity)
        predicted_time, predicted_landing, _, _, reached = predict_land(
            hit_pos,
            desired_velocity,
            target_spin,
            landing_h=landing_pos[..., 2],
            max_time=1.2,
            event_root_iterations=event_root_iterations,
            event_integration_substeps=event_integration_substeps,
            physics=physics,
        )
        valid &= reached
        if iteration + 1 < correction_iterations:
            perturbed_velocity = desired_velocity.unsqueeze(-2).expand(*desired_velocity.shape[:-1], 2, 3).clone()
            perturbed_velocity[..., 0, 0] += perturbation
            perturbed_velocity[..., 1, 1] += perturbation
            flat_velocity = perturbed_velocity.reshape(-1, 3)
            flat_spin_command = spin_command.unsqueeze(-2).expand(
                *spin_command.shape[:-1], 2, 3
            ).reshape(-1, 3)
            flat_hit_position = hit_pos.unsqueeze(-2).expand(
                *hit_pos.shape[:-1], 2, 3
            ).reshape(-1, 3)
            flat_landing_height = landing_pos[..., 2].unsqueeze(-1).expand(
                *landing_pos.shape[:-1], 2
            ).reshape(-1)
            flat_spin = trajectory_spin_to_world(flat_spin_command, flat_velocity)
            _, perturbed_landing, _, _, _ = predict_land(
                flat_hit_position,
                flat_velocity,
                flat_spin,
                landing_h=flat_landing_height,
                max_time=1.2,
                event_root_iterations=event_root_iterations,
                event_integration_substeps=event_integration_substeps,
                physics=physics,
            )
            desired_velocity[..., :2] += _horizontal_newton_update(
                predicted_landing,
                landing_pos,
                perturbed_landing.reshape(*hit_pos.shape[:-1], 2, 3),
                perturbation,
            )

    target_spin = trajectory_spin_to_world(spin_command, desired_velocity)
    _, net_position, _, _, crossed_net = propagate_to_x(
        hit_pos,
        desired_velocity,
        target_spin,
        net_x,
        max_time=1.0,
        params=physics,
        root_iterations=event_root_iterations,
        integration_substeps=event_integration_substeps,
    )
    clears_net = crossed_net & (net_position[..., 2] >= net_clearance)
    return desired_velocity, target_spin, predicted_landing, valid & clears_net


def solve_racket_command(
    incoming_velocity: torch.Tensor,
    incoming_spin: torch.Tensor,
    desired_out_velocity: torch.Tensor,
    desired_out_spin: torch.Tensor,
    *,
    spin_priority: float = 1.0,
    physics: BallPhysicsParams = DEFAULT_BALL_PHYSICS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Invert the analytic impact approximately into a racket twist.

    A single tangential impulse couples outgoing velocity and spin, so arbitrary
    pairs are not always simultaneously feasible.  The solver selects the
    racket normal analytically so the desired velocity remains exact while the
    achievable component of the requested spin change is maximized.
    """

    delta_velocity = desired_out_velocity - incoming_velocity
    delta_speed_sq = torch.sum(delta_velocity**2, dim=-1, keepdim=True).clamp_min(1.0e-8)
    delta_direction = delta_velocity / torch.sqrt(delta_speed_sq)

    # For the thin-shell impact model, tangential impulse imposes
    # delta_spin = -(kappa / radius) normal x delta_velocity_tangent.
    # Solve normal x delta_velocity = -(radius / kappa) delta_spin.
    requested_delta_spin = spin_priority * (desired_out_spin - incoming_spin)
    cross_target = -physics.radius / physics.inertia_ratio * requested_delta_spin
    cross_target -= (
        torch.sum(cross_target * delta_velocity, dim=-1, keepdim=True)
        / delta_speed_sq
        * delta_velocity
    )
    normal_perpendicular = torch.cross(delta_velocity, cross_target, dim=-1) / delta_speed_sq
    perpendicular_norm = torch.linalg.vector_norm(normal_perpendicular, dim=-1, keepdim=True)
    # The requested tangential/normal impulse ratio must fit inside the racket
    # Coulomb cone.  This replaces the previous near-90-degree normal, which
    # could request almost free spin on a grazing contact.
    max_perpendicular = physics.racket_friction / math.sqrt(1.0 + physics.racket_friction**2)
    max_perpendicular *= 0.995
    normal_perpendicular = normal_perpendicular / torch.maximum(
        torch.ones_like(perpendicular_norm), perpendicular_norm / max_perpendicular
    )
    parallel_scale = torch.sqrt(
        torch.clamp(1.0 - torch.sum(normal_perpendicular**2, dim=-1, keepdim=True), min=1.0e-6)
    )
    normal = normal_perpendicular + parallel_scale * delta_direction
    normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(1.0e-8)

    desired_normal_delta = torch.sum(delta_velocity * normal, dim=-1).clamp_min(1.0e-4)
    normal_speed = desired_normal_delta / (1.0 + physics.racket_normal_restitution_base)
    for _ in range(3):
        restitution = (
            physics.racket_normal_restitution_base
            + physics.racket_normal_restitution_velocity_slope * normal_speed
        ).clamp(physics.racket_normal_restitution_min, physics.racket_normal_restitution_max)
        normal_speed = desired_normal_delta / (1.0 + restitution)
    relative_normal_velocity = -normal_speed

    delta_velocity_tangent = delta_velocity - desired_normal_delta.unsqueeze(-1) * normal
    alpha = torch.full_like(normal_speed, (1.0 + physics.racket_tangential_restitution_base) / (1.0 + physics.inertia_ratio))
    relative_tangent_from_velocity = -delta_velocity_tangent / alpha.unsqueeze(-1).clamp_min(1.0e-6)
    relative_tangent = relative_tangent_from_velocity

    for _ in range(3):
        tangential_speed = torch.linalg.vector_norm(relative_tangent, dim=-1)
        tangential_restitution = (
            physics.racket_tangential_restitution_base
            + physics.racket_tangential_restitution_velocity_slope * tangential_speed
        ).clamp(physics.racket_tangential_restitution_min, physics.racket_tangential_restitution_max)
        alpha = (1.0 + tangential_restitution) / (1.0 + physics.inertia_ratio)
        relative_tangent_from_velocity = -delta_velocity_tangent / alpha.unsqueeze(-1).clamp_min(1.0e-6)
        relative_tangent = relative_tangent_from_velocity

    ball_contact_velocity = incoming_velocity + physics.radius * torch.cross(normal, incoming_spin, dim=-1)
    racket_contact_velocity = ball_contact_velocity - (
        relative_normal_velocity.unsqueeze(-1) * normal + relative_tangent
    )
    racket_spin = torch.zeros_like(racket_contact_velocity)
    predicted_velocity, predicted_spin, impact_info = racket_impact(
        incoming_velocity,
        incoming_spin,
        racket_contact_velocity,
        racket_spin,
        normal,
        params=physics,
    )
    return racket_contact_velocity, racket_spin, impact_info["normal"], predicted_velocity, predicted_spin


def split_contact_velocity_into_racket_twist(
    contact_velocity: torch.Tensor,
    contact_offset: torch.Tensor,
    *,
    wrist_fraction: float = 0.35,
    max_angular_speed: float = 30.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a desired contact-point velocity into body translation and spin.

    The split is underdetermined.  A configurable fraction of the component
    perpendicular to the handle-to-contact offset is assigned to angular
    velocity, then the body velocity is the exact residual.  Consequently
    ``v_body + omega x offset`` always equals ``contact_velocity``.
    """

    offset_norm_sq = torch.sum(contact_offset**2, dim=-1, keepdim=True).clamp_min(1.0e-8)
    parallel = (
        torch.sum(contact_velocity * contact_offset, dim=-1, keepdim=True)
        / offset_norm_sq
        * contact_offset
    )
    rotational_contribution = wrist_fraction * (contact_velocity - parallel)
    angular_velocity = torch.cross(contact_offset, rotational_contribution, dim=-1) / offset_norm_sq
    angular_speed = torch.linalg.vector_norm(angular_velocity, dim=-1, keepdim=True)
    angular_velocity *= torch.minimum(
        torch.ones_like(angular_speed),
        torch.full_like(angular_speed, max_angular_speed) / angular_speed.clamp_min(1.0e-8),
    )
    body_velocity = contact_velocity - torch.cross(angular_velocity, contact_offset, dim=-1)
    return body_velocity, angular_velocity


def plan_spin_return(
    incoming: IncomingHitPrediction,
    target_landing_position: torch.Tensor,
    spin_command: torch.Tensor,
    *,
    paddle_face_local: torch.Tensor | None = None,
    spin_priority: float = 1.0,
    shot_correction_iterations: int = 2,
    impact_correction_iterations: int = 3,
    wrist_fraction: float = 0.35,
    max_paddle_speed: float = 12.0,
    max_paddle_angular_speed: float = 30.0,
    event_root_iterations: int = 4,
    event_integration_substeps: int = 3,
    physics: BallPhysicsParams = DEFAULT_BALL_PHYSICS,
) -> SpinReturnPlan:
    """Plan a legal return conditioned on landing position and requested spin."""

    desired_velocity, desired_spin, _, valid_shot = solve_outgoing_velocity(
        incoming.position,
        target_landing_position,
        spin_command,
        correction_iterations=shot_correction_iterations,
        event_root_iterations=event_root_iterations,
        event_integration_substeps=event_integration_substeps,
        physics=physics,
    )
    valid = incoming.valid & valid_shot
    if impact_correction_iterations < 1:
        raise ValueError("impact_correction_iterations must be positive")
    predicted_landing = target_landing_position.clone()
    contact_velocity = torch.zeros_like(incoming.velocity)
    normal = torch.zeros_like(incoming.velocity)
    predicted_velocity = desired_velocity
    predicted_spin = desired_spin
    perturbation = 0.05
    for iteration in range(impact_correction_iterations):
        contact_velocity, _, normal, predicted_velocity, predicted_spin = solve_racket_command(
            incoming.velocity,
            incoming.spin,
            desired_velocity,
            desired_spin,
            spin_priority=spin_priority,
            physics=physics,
        )
        landing_time, predicted_landing, _, _, valid_landing = predict_land(
            incoming.position,
            predicted_velocity,
            predicted_spin,
            landing_h=target_landing_position[..., 2],
            max_time=1.2,
            event_root_iterations=event_root_iterations,
            event_integration_substeps=event_integration_substeps,
            physics=physics,
        )
        valid &= valid_landing
        if iteration + 1 < impact_correction_iterations:
            perturbed_velocity = desired_velocity.unsqueeze(-2).expand(
                *desired_velocity.shape[:-1], 2, 3
            ).clone()
            perturbed_velocity[..., 0, 0] += perturbation
            perturbed_velocity[..., 1, 1] += perturbation
            flat_velocity = perturbed_velocity.reshape(-1, 3)
            flat_spin_command = spin_command.unsqueeze(-2).expand(
                *spin_command.shape[:-1], 2, 3
            ).reshape(-1, 3)
            flat_desired_spin = trajectory_spin_to_world(flat_spin_command, flat_velocity)
            flat_incoming_velocity = incoming.velocity.unsqueeze(-2).expand(
                *incoming.velocity.shape[:-1], 2, 3
            ).reshape(-1, 3)
            flat_incoming_spin = incoming.spin.unsqueeze(-2).expand(
                *incoming.spin.shape[:-1], 2, 3
            ).reshape(-1, 3)
            _, _, _, flat_predicted_velocity, flat_predicted_spin = solve_racket_command(
                flat_incoming_velocity,
                flat_incoming_spin,
                flat_velocity,
                flat_desired_spin,
                spin_priority=spin_priority,
                physics=physics,
            )
            flat_hit_position = incoming.position.unsqueeze(-2).expand(
                *incoming.position.shape[:-1], 2, 3
            ).reshape(-1, 3)
            flat_landing_height = target_landing_position[..., 2].unsqueeze(-1).expand(
                *target_landing_position.shape[:-1], 2
            ).reshape(-1)
            _, flat_perturbed_landing, _, _, _ = predict_land(
                flat_hit_position,
                flat_predicted_velocity,
                flat_predicted_spin,
                landing_h=flat_landing_height,
                max_time=1.2,
                event_root_iterations=event_root_iterations,
                event_integration_substeps=event_integration_substeps,
                physics=physics,
            )
            desired_velocity[..., :2] += _horizontal_newton_update(
                predicted_landing,
                target_landing_position,
                flat_perturbed_landing.reshape(*incoming.position.shape[:-1], 2, 3),
                perturbation,
            )
            desired_spin = trajectory_spin_to_world(spin_command, desired_velocity)
    if paddle_face_local is None:
        paddle_face_local = torch.zeros_like(normal)
        paddle_face_local[..., 2] = -1.0
    paddle_orientation = vec_to_quat(paddle_face_local, normal)
    paddle_face_center = incoming.position - (
        physics.radius + DEFAULT_PADDLE_HALF_THICKNESS
    ) * normal
    paddle_position = compute_paddle_pos(paddle_orientation, paddle_face_center)
    pad_center_offset = paddle_face_center - paddle_position
    contact_offset = pad_center_offset + DEFAULT_PADDLE_HALF_THICKNESS * normal
    # `contact_offset` rotates with the paddle, so express it in the body frame
    # once here: `pad_center_offset` is `R @ PAD_CENTER_OFFSET_LOCAL` and
    # `normal` is `R @ paddle_face_local`, hence the local form below.
    contact_offset_local = (
        torch.tensor(
            PAD_CENTER_OFFSET_LOCAL, dtype=normal.dtype, device=normal.device
        ).expand_as(paddle_face_local)
        + DEFAULT_PADDLE_HALF_THICKNESS * paddle_face_local
    )
    paddle_velocity, paddle_spin = split_contact_velocity_into_racket_twist(
        contact_velocity,
        contact_offset,
        wrist_fraction=wrist_fraction,
        max_angular_speed=max_paddle_angular_speed,
    )
    predicted_velocity, predicted_spin, _ = racket_impact(
        incoming.velocity,
        incoming.spin,
        paddle_velocity,
        paddle_spin,
        normal,
        contact_offset,
        params=physics,
    )
    _, net_position, _, _, crossed_net = propagate_to_x(
        incoming.position,
        predicted_velocity,
        predicted_spin,
        0.0,
        max_time=1.0,
        params=physics,
        root_iterations=event_root_iterations,
        integration_substeps=event_integration_substeps,
    )
    valid &= crossed_net & (net_position[..., 2] >= MIN_RETURN_NET_HEIGHT)
    valid &= torch.linalg.vector_norm(paddle_velocity, dim=-1) <= max_paddle_speed
    valid &= torch.linalg.vector_norm(paddle_spin, dim=-1) <= max_paddle_angular_speed + 1.0e-5
    return SpinReturnPlan(
        paddle_position=paddle_position,
        paddle_velocity=paddle_velocity,
        paddle_spin=paddle_spin,
        paddle_orientation=paddle_orientation,
        contact_velocity=contact_velocity,
        contact_offset_local=contact_offset_local,
        hit_position=incoming.position,
        hit_time=incoming.time,
        target_landing_position=target_landing_position,
        predicted_out_velocity=predicted_velocity,
        predicted_out_spin=predicted_spin,
        predicted_landing_position=predicted_landing,
        valid=valid & valid_landing,
    )


def compute_land(ball_pos, ball_vel, landing_h=0.795):
    g = -9.81
    discriminant = ball_vel[:, 2] ** 2 - 2 * g * (ball_pos[:, 2] - landing_h)
    t_land = (-ball_vel[:, 2] - torch.sqrt(discriminant.clamp_min(0.0))) / g

    land_pos = torch.zeros_like(ball_pos)
    land_pos[:, 0:2] = ball_pos[:, 0:2] + ball_vel[:, 0:2] * t_land.unsqueeze(-1)
    land_pos[:, 2] = ball_pos[:, 2] + ball_vel[:, 2] * t_land + 0.5 * g * t_land**2
    land_vel = torch.zeros_like(ball_vel)
    land_vel[:, 0:2] = ball_vel[:, 0:2]
    land_vel[:, 2] = ball_vel[:, 2] + g * t_land
    return t_land, land_pos, land_vel


def compute_hit_pos(t_land, bounce_pos, bounce_vel, hit_plane_x=1.8):
    g = -9.81
    t_hit = (hit_plane_x - bounce_pos[:, 0]) / bounce_vel[:, 0]
    t_total_round = torch.round(t_land + t_hit, decimals=2)
    t_hit_round = t_total_round - t_land

    hit_pos = torch.zeros_like(bounce_pos)
    hit_pos[:, 0:2] = bounce_pos[:, 0:2] + bounce_vel[:, 0:2] * t_hit_round.unsqueeze(-1)
    hit_pos[:, 2] = 0.5 * g * t_hit_round**2 + bounce_vel[:, 2] * t_hit_round + bounce_pos[:, 2]

    hit_vel = torch.zeros_like(bounce_vel)
    hit_vel[:, 0:2] = bounce_vel[:, 0:2]
    hit_vel[:, 2] = bounce_vel[:, 2] + g * t_hit_round

    return hit_pos, hit_vel, t_total_round


def compute_paddle_vel(hit_pos, v_in, table_area_upper, table_area_lower, device, C_r=1.0, net_h=0.95+0.05, net_x=0.0):
    # TODO: resample until v_out lands within the target serving area
    n_envs = hit_pos.shape[0]
    gravity = 9.81

    # sample V_z
    v_z = torch.rand((n_envs,)).to(device) * 0.2 + 0.6

    # compute t
    a = -0.5 * gravity
    b = v_z
    c = hit_pos[:, 2] - table_area_upper[2]

    discriminant = b**2 - 4 * a * c
    t = (-b - discriminant**0.5) / (2 * a)

    # compute V_x and V_y
    v_upper = torch.stack(
            [(table_area_upper[0] - hit_pos[:, 0]) / t, (table_area_upper[1] - hit_pos[:, 1]) / t, v_z], dim=-1
        )
    v_lower = torch.stack(
            [(table_area_lower[0] - hit_pos[:, 0]) / t, (table_area_lower[1] - hit_pos[:, 1]) / t, v_z], dim=-1
        )
    # TODO: v_x has a lower bound - the ball must clear the net
    A = net_h - hit_pos[:, 2]
    B = -v_z * (net_x - hit_pos[:, 0])
    C = 0.5 * gravity * (net_x - hit_pos[:, 0])**2
    v_x_min = find_low_v_x(A, B, C)

    v_lower[:,0] = torch.minimum(v_lower[:,0], v_x_min)

    # Resample until all trajectories reach the opponent's hit plane
    v_out = torch.rand((n_envs, 3)).to(device) * (v_upper - v_lower) + v_lower
    valid_mask = torch.zeros(n_envs, device=device, dtype=torch.bool)

    max_iterations = 50
    iteration = 0

    while not valid_mask.all() and iteration < max_iterations:
        t_land, land_pos, land_vel = compute_land(hit_pos, v_out)
        bounce_vel = land_vel.clone()
        bounce_vel[:, 2] = -bounce_vel[:, 2]
        hit_plane_pos, _, _ = compute_hit_pos(t_land, land_pos, bounce_vel, hit_plane_x=-1.8)

        valid_mask = (hit_plane_pos[:, 2] > 1.1) & (hit_plane_pos[:, 2] < 1.35) & (hit_plane_pos[:, 1] > -0.7) & (hit_plane_pos[:, 1] < 0.7)

        if not valid_mask.all():
            invalid_indices = ~valid_mask
            n_invalid = invalid_indices.sum().item()
            if n_invalid > 0:
                v_z_new = torch.rand((n_invalid,)).to(device) * 0.2 + 0.6
                b = v_z_new
                c = hit_pos[invalid_indices, 2] - table_area_upper[2]

                discriminant = b**2 - 4 * a * c
                t = (-b - discriminant**0.5) / (2 * a)

                v_upper = torch.stack(
                        [(table_area_upper[0] - hit_pos[invalid_indices, 0]) / t, (table_area_upper[1] - hit_pos[invalid_indices, 1]) / t, v_z_new], dim=-1
                    )
                v_lower = torch.stack(
                        [(table_area_lower[0] - hit_pos[invalid_indices, 0]) / t, (table_area_lower[1] - hit_pos[invalid_indices, 1]) / t, v_z_new], dim=-1
                    )
                # v_lower.shape[0] = len(invalid_indices)

                A = net_h - hit_pos[invalid_indices, 2]
                B = -v_z_new * (net_x - hit_pos[invalid_indices, 0])
                C = 0.5 * gravity * (net_x - hit_pos[invalid_indices, 0])**2
                v_x_min = find_low_v_x(A, B, C)

                v_lower[:, 0] = torch.minimum(v_lower[:, 0], v_x_min)

                v_out_new = torch.rand((n_invalid, 3)).to(device) * (v_upper - v_lower) + v_lower
                v_out[invalid_indices] = v_out_new

        iteration += 1

    # Collision normal (unit vector pointing in the direction of velocity change):
    # u = (v_out - v_in) / ||v_out - v_in||
    u = (v_out - v_in) / torch.norm(v_out - v_in, dim=-1, keepdim=True)

    # Paddle velocity from elastic collision model:
    # v_racket = (dot(v_out, u) + dot(v_in, u)) / (1 + C_r) * u
    v_racket = (torch.sum(v_out * u, dim=-1, keepdim=True) + torch.sum(v_in * u, dim=-1, keepdim=True)) / (1 + C_r) * u

    # TODO: add small randomization to improve robustness

    return v_racket


def find_low_v_x(A, B, C):
    delta = B**2 - 4 * A * C

    if (delta < 0).any():
        mask_delta_neg = delta < 0
        root1 = (-B + delta**0.5) / (2 * A)
        root2 = torch.zeros_like(root1)
        return torch.where(mask_delta_neg, root2, root1)

    root1 = (-B + delta**0.5) / (2 * A)
    return root1
    



def vec_to_quat(from_vec, to_vec):
    """
    Compute the quaternion that rotates from_vec to to_vec.

    Computes an absolute rotation quaternion, starting from the object's
    unrotated pose (local and global frames coincide).

    Args:
        from_vec: (n, 3) tensor
        to_vec: (n, 3) tensor

    Returns:
        quat: (n, 4) tensor in (w, x, y, z) format
    """
    from_vec = from_vec / torch.norm(from_vec, dim=-1, keepdim=True)
    to_vec = to_vec / torch.norm(to_vec, dim=-1, keepdim=True)

    axis = torch.cross(from_vec, to_vec, dim=-1)
    angle = torch.acos(torch.clamp(torch.sum(from_vec * to_vec, dim=-1), -1.0, 1.0))

    axis_norm = torch.norm(axis, dim=-1, keepdim=True)

    # Nearly parallel (axis_norm < 1e-6 and angle < 0.1): no rotation
    parallel_mask = (axis_norm.squeeze(-1) < 1e-6) & (angle < 0.1)
    # Anti-parallel (axis_norm < 1e-6 and angle >= 0.1): 180° rotation about X
    opposite_mask = (axis_norm.squeeze(-1) < 1e-6) & (angle >= 0.1)
    normal_mask = ~(parallel_mask | opposite_mask)

    axis_normalized = axis / (axis_norm + 1e-8)

    half_angle = angle / 2
    quat = torch.zeros(from_vec.shape[0], 4, device=from_vec.device, dtype=from_vec.dtype)

    quat[normal_mask, 0] = torch.cos(half_angle[normal_mask])
    quat[normal_mask, 1:4] = axis_normalized[normal_mask] * torch.sin(half_angle[normal_mask]).unsqueeze(-1)
    quat[parallel_mask, 0] = 1.0   # identity
    quat[opposite_mask, 1] = 1.0   # 180° about X-axis

    return quat


def compute_paddle_pos(paddle_ori, hit_pos):
    # get paddle pos
    # paddle_ori is (n, 4) tensor with format [w, x, y, z]
    # Convert quaternion to rotation matrix
    w, x, y, z = paddle_ori[:, 0], paddle_ori[:, 1], paddle_ori[:, 2], paddle_ori[:, 3]
    
    # Quaternion to rotation matrix conversion
    rot_mat = torch.zeros(paddle_ori.shape[0], 3, 3, device=paddle_ori.device, dtype=paddle_ori.dtype)
    rot_mat[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    rot_mat[:, 0, 1] = 2 * (x*y - w*z)
    rot_mat[:, 0, 2] = 2 * (x*z + w*y)
    rot_mat[:, 1, 0] = 2 * (x*y + w*z)
    rot_mat[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    rot_mat[:, 1, 2] = 2 * (y*z - w*x)
    rot_mat[:, 2, 0] = 2 * (x*z - w*y)
    rot_mat[:, 2, 1] = 2 * (y*z + w*x)
    rot_mat[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    
    # Local paddle offset
    pad_local = torch.tensor(
        PAD_CENTER_OFFSET_LOCAL, device=paddle_ori.device, dtype=paddle_ori.dtype
    )
    # Batch matrix multiplication: (n, 3, 3) @ (3,) -> (n, 3)
    pad_world = torch.matmul(rot_mat, pad_local)
    p_paddle = hit_pos - pad_world    

    return p_paddle


def compute_land_net(ball_pos, ball_vel, net_x=0.0, net_h=0.95):
    g = -9.81
    t_land_net = (net_x - ball_pos[:, 0]) / ball_vel[:, 0]
    land_pos = torch.zeros_like(ball_pos)
    land_pos[:, 0:2] = ball_pos[:, 0:2] + ball_vel[:, 0:2] * t_land_net.unsqueeze(-1)
    land_pos[:, 2] = ball_pos[:, 2] + ball_vel[:, 2] * t_land_net + 0.5 * g * t_land_net**2
    
    # Ball is out of bounds (y outside table width) or hits the net (z below net height)
    fail = torch.zeros_like(ball_pos[:, 1], dtype=torch.bool)
    fail |= (land_pos[:, 1] > 0.80)
    fail |= (land_pos[:, 1] < -0.72)
    fail |= (land_pos[:, 2] < net_h)

    return fail
