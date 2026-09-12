"""Regression tests for the batched spin-aware paddle planner."""

import torch
from ball_physics import (
    propagate_to_height_reference,
    propagate_to_x,
    trajectory_spin_to_world,
    world_spin_to_trajectory,
)
from planner import (
    IncomingHitPrediction,
    plan_spin_return,
    predict_incoming_hit,
    split_contact_velocity_into_racket_twist,
)


def _quat_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quaternion_vector = quaternion[..., 1:]
    twice_cross = 2.0 * torch.cross(quaternion_vector, vector, dim=-1)
    return vector + quaternion[..., :1] * twice_cross + torch.cross(
        quaternion_vector, twice_cross, dim=-1
    )


def _incoming_batch():
    position = torch.tensor(
        [
            [-1.8, 0.0, 1.20],
            [-1.8, 0.2, 1.30],
            [-1.8, -0.2, 1.25],
        ]
    )
    velocity = torch.tensor(
        [
            [7.0, 0.0, 0.8],
            [6.5, -0.5, 0.7],
            [7.5, 0.4, 0.6],
        ]
    )
    incoming_command = torch.tensor(
        [
            [100.0, 30.0, 0.0],
            [-150.0, 0.0, 0.0],
            [0.0, -80.0, 0.0],
        ]
    )
    spin = trajectory_spin_to_world(incoming_command, velocity)
    return predict_incoming_hit(position, velocity, spin)


def test_incoming_prediction_reaches_robot_hit_plane():
    incoming = _incoming_batch()

    assert torch.equal(incoming.valid, torch.tensor([False, True, True]))
    assert torch.allclose(incoming.position[:, 0], torch.full((3,), 1.8), atol=1.0e-6)
    assert torch.isfinite(incoming.position).all()
    assert torch.isfinite(incoming.spin).all()


def test_return_plan_tracks_landing_and_spin_commands():
    incoming = _incoming_batch()
    landing_target = torch.tensor(
        [
            [-0.9, 0.0, 0.815],
            [-1.0, -0.2, 0.815],
            [-0.8, 0.25, 0.815],
        ]
    )
    spin_command = torch.tensor(
        [
            [300.0, 0.0, 0.0],
            [-250.0, 100.0, 0.0],
            [0.0, -180.0, 0.0],
        ]
    )

    plan = plan_spin_return(incoming, landing_target, spin_command)
    achieved_spin = world_spin_to_trajectory(plan.predicted_out_spin, plan.predicted_out_velocity)
    landing_error = torch.linalg.vector_norm(
        plan.predicted_landing_position[:, :2] - landing_target[:, :2], dim=-1
    )

    # The first fixture reaches the robot plane below table height and is now
    # correctly rejected instead of being planned through a second bounce.
    assert torch.equal(plan.valid, torch.tensor([False, True, True]))
    assert torch.all(landing_error[plan.valid] < 0.03)
    assert torch.all(
        torch.abs(achieved_spin[plan.valid, :2] - spin_command[plan.valid, :2]) < 15.0
    )
    assert torch.isfinite(plan.paddle_velocity).all()
    assert torch.isfinite(plan.paddle_orientation).all()
    assert torch.linalg.vector_norm(plan.paddle_spin[1:], dim=-1).min() > 0.0

    local_face_normal = torch.tensor([0.0, 0.0, -1.0]).expand(3, -1)
    local_pad_center = torch.tensor([-0.07, 0.0, 0.0]).expand(3, -1)
    world_face_normal = _quat_rotate(plan.paddle_orientation, local_face_normal)
    world_pad_center = plan.paddle_position + _quat_rotate(plan.paddle_orientation, local_pad_center)
    center_offset = world_pad_center - plan.hit_position
    assert torch.allclose(
        center_offset,
        -0.04 * world_face_normal,
        atol=1.0e-5,
        rtol=0.0,
    )


def test_return_planner_monte_carlo_regression():
    torch.manual_seed(19)
    batch_size = 128
    hit_position = torch.column_stack(
        (
            torch.full((batch_size,), 1.8),
            torch.empty(batch_size).uniform_(-0.35, 0.35),
            torch.empty(batch_size).uniform_(1.12, 1.32),
        )
    )
    incoming_velocity = torch.column_stack(
        (
            torch.empty(batch_size).uniform_(5.5, 8.0),
            torch.empty(batch_size).uniform_(-0.5, 0.5),
            torch.empty(batch_size).uniform_(0.2, 2.0),
        )
    )
    incoming_command = torch.column_stack(
        (
            torch.empty(batch_size).uniform_(-300.0, 300.0),
            torch.empty(batch_size).uniform_(-200.0, 200.0),
            torch.empty(batch_size).uniform_(-80.0, 80.0),
        )
    )
    incoming = IncomingHitPrediction(
        position=hit_position,
        velocity=incoming_velocity,
        spin=trajectory_spin_to_world(incoming_command, incoming_velocity),
        time=torch.empty(batch_size).uniform_(0.4, 0.8),
        valid=torch.ones(batch_size, dtype=torch.bool),
    )
    landing_target = torch.column_stack(
        (
            torch.empty(batch_size).uniform_(-1.3, -0.55),
            torch.empty(batch_size).uniform_(-0.35, 0.35),
            torch.full((batch_size,), 0.815),
        )
    )
    spin_command = torch.column_stack(
        (
            torch.empty(batch_size).uniform_(-350.0, 350.0),
            torch.empty(batch_size).uniform_(-220.0, 220.0),
            torch.zeros(batch_size),
        )
    )

    plan = plan_spin_return(incoming, landing_target, spin_command)
    achieved_spin = world_spin_to_trajectory(plan.predicted_out_spin, plan.predicted_out_velocity)
    landing_error = torch.linalg.vector_norm(
        plan.predicted_landing_position[:, :2] - landing_target[:, :2], dim=-1
    )
    controlled_spin_error = torch.abs(achieved_spin[:, :2] - spin_command[:, :2])

    assert plan.valid.float().mean() > 0.98
    assert torch.quantile(landing_error, 0.95) < 0.002
    assert torch.quantile(controlled_spin_error, 0.95) < 12.0
    assert torch.isfinite(plan.paddle_velocity).all()
    _, net_position, _, _, crossed_net = propagate_to_x(
        incoming.position, plan.predicted_out_velocity, plan.predicted_out_spin, 0.0
    )
    assert crossed_net[plan.valid].all()
    assert torch.all(net_position[plan.valid, 2] >= 1.0)
    assert torch.all(torch.linalg.vector_norm(plan.paddle_velocity[plan.valid], dim=-1) <= 12.0)
    assert torch.all(torch.linalg.vector_norm(plan.paddle_spin[plan.valid], dim=-1) <= 30.0)


def test_planned_shot_lands_on_target_under_fine_integration():
    """Check the plan against fine integration, not against its own predictor.

    The Monte-Carlo test above compares ``predicted_landing_position`` to the
    target, so it passes whenever the planner is self-consistent even if its
    fast event solver drifts.  Re-integrating the planned outgoing state at
    0.5 ms closes that loop, and pins the two iteration counts that drive it:
    dropping ``shot_correction_iterations`` to 1 collapses the valid fraction
    (no trajectory clears the net), and dropping
    ``impact_correction_iterations`` to 1 leaves a ~230 mm landing error.
    """

    generator = torch.Generator().manual_seed(19)
    batch_size = 512
    hit_position = torch.column_stack(
        (
            torch.full((batch_size,), 1.8),
            torch.empty(batch_size).uniform_(-0.35, 0.35, generator=generator),
            torch.empty(batch_size).uniform_(1.12, 1.32, generator=generator),
        )
    )
    incoming_velocity = torch.column_stack(
        (
            torch.empty(batch_size).uniform_(5.5, 8.0, generator=generator),
            torch.empty(batch_size).uniform_(-0.5, 0.5, generator=generator),
            torch.empty(batch_size).uniform_(0.2, 2.0, generator=generator),
        )
    )
    incoming_command = torch.column_stack(
        (
            torch.empty(batch_size).uniform_(-350.0, 350.0, generator=generator),
            torch.empty(batch_size).uniform_(-220.0, 220.0, generator=generator),
            torch.zeros(batch_size),
        )
    )
    incoming = IncomingHitPrediction(
        position=hit_position,
        velocity=incoming_velocity,
        spin=trajectory_spin_to_world(incoming_command, incoming_velocity),
        time=torch.zeros(batch_size),
        valid=torch.ones(batch_size, dtype=torch.bool),
    )
    landing_target = torch.column_stack(
        (
            torch.empty(batch_size).uniform_(-1.3, -0.55, generator=generator),
            torch.empty(batch_size).uniform_(-0.35, 0.35, generator=generator),
            torch.full((batch_size,), 0.815),
        )
    )
    # Sweep the commanded topspin so heavy backspin, which flies longest and is
    # the hardest case for the event solver, is always represented.
    spin_command = torch.column_stack(
        (
            torch.linspace(-400.0, 400.0, batch_size),
            torch.empty(batch_size).uniform_(-250.0, 250.0, generator=generator),
            torch.zeros(batch_size),
        )
    )

    plan = plan_spin_return(incoming, landing_target, spin_command)
    _, fine_landing, _, _, reached = propagate_to_height_reference(
        incoming.position,
        plan.predicted_out_velocity,
        plan.predicted_out_spin,
        0.815,
        dt=0.0005,
        max_time=1.5,
    )

    assert plan.valid.float().mean() > 0.99
    evaluated = plan.valid & reached
    assert evaluated.float().mean() > 0.99
    true_error = torch.linalg.vector_norm(
        fine_landing[evaluated][:, :2] - landing_target[evaluated][:, :2], dim=-1
    )
    assert torch.quantile(true_error, 0.95) < 5.0e-3
    assert true_error.max() < 4.0e-2

    # A single tangential impulse cannot reach an arbitrary (velocity, spin)
    # pair: the spin budget is bounded by (kappa / r) * |delta v|.  Commands
    # well inside that budget must be tracked tightly and symmetrically in
    # both spin directions; larger ones fall short in magnitude while keeping
    # the right axis, which is checked separately below.
    achieved = world_spin_to_trajectory(plan.predicted_out_spin, plan.predicted_out_velocity)
    spin_error = (achieved[:, :2] - spin_command[:, :2]).abs().amax(dim=-1)
    reachable = torch.linalg.vector_norm(spin_command[:, :2], dim=-1) <= 250.0
    backspin = reachable & (spin_command[:, 0] < -100.0)
    topspin = reachable & (spin_command[:, 0] > 100.0)
    assert backspin.sum() > 20 and topspin.sum() > 20
    assert torch.quantile(spin_error[backspin], 0.95) < 12.0
    assert torch.quantile(spin_error[topspin], 0.95) < 12.0

    # Even where the commanded magnitude is not attainable, the spin axis must
    # stay correct, so the shortfall never inverts the requested effect.
    target_world = trajectory_spin_to_world(spin_command, plan.predicted_out_velocity)
    cosine = torch.sum(plan.predicted_out_spin * target_world, dim=-1) / (
        plan.predicted_out_spin.norm(dim=-1) * target_world.norm(dim=-1)
    ).clamp_min(1.0e-8)
    axis_error = torch.rad2deg(torch.arccos(cosine.clamp(-1.0, 1.0)))
    assert torch.quantile(axis_error, 0.95) < 20.0
    assert axis_error.max() < 45.0


def test_racket_twist_preserves_contact_point_velocity():
    contact_velocity = torch.tensor([[-4.0, 1.0, 3.0], [-5.0, -0.5, 2.0]])
    contact_offset = torch.tensor([[-0.07, 0.0, 0.02], [-0.06, 0.01, -0.02]])
    body_velocity, angular_velocity = split_contact_velocity_into_racket_twist(
        contact_velocity, contact_offset
    )

    reconstructed = body_velocity + torch.cross(angular_velocity, contact_offset, dim=-1)
    assert torch.allclose(reconstructed, contact_velocity, atol=1.0e-6)
    assert torch.linalg.vector_norm(angular_velocity, dim=-1).min() > 0.0
