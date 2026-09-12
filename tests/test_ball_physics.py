"""Focused tests for spin-aware table-tennis ball physics."""

from dataclasses import replace

import pytest
import torch
from ball_physics import (
    DEFAULT_BALL_PHYSICS,
    aerodynamic_force,
    aerodynamic_force_components,
    drag_coefficient,
    integrate_flight,
    magnus_coefficient,
    propagate_to_height,
    propagate_to_height_reference,
    propagate_to_x,
    propagate_to_x_reference,
    racket_impact,
    table_impact,
    trajectory_spin_to_world,
    world_spin_to_trajectory,
)


def test_ball_shell_inertia_matches_geometry():
    assert DEFAULT_BALL_PHYSICS.shell_inertia == pytest.approx(7.2e-7)
    assert DEFAULT_BALL_PHYSICS.inertia_ratio == pytest.approx(1.5)


def test_constant_model_reproduces_tt4d_force_coefficients():
    params = DEFAULT_BALL_PHYSICS
    drag_force_coefficient = 0.5 * params.air_density * params.area * params.drag_coefficient
    magnus_force_coefficient = params.air_density * params.volume * params.magnus_coefficient

    assert drag_force_coefficient == pytest.approx(3.8e-4, rel=1.0e-6)
    assert magnus_force_coefficient == pytest.approx(3.0e-6, rel=1.0e-6)


def test_drag_dissipates_and_magnus_does_no_work():
    velocity = torch.tensor([[8.0, -1.0, 2.0]])
    spin = torch.tensor([[0.0, 300.0, 150.0]])
    drag, magnus = aerodynamic_force_components(velocity, spin)

    assert torch.sum(drag * velocity) < 0.0
    assert torch.sum(magnus * velocity).abs() < 1.0e-7


def test_fitted_aerodynamic_coefficients_are_finite_and_speed_sensitive():
    velocity = torch.tensor([[2.0, 0.0, 0.0], [8.0, 0.0, 0.0], [17.0, 0.0, 0.0]])
    spin = torch.tensor([[0.0, 300.0, 0.0]]).expand_as(velocity)

    c_drag = drag_coefficient(velocity, spin)
    c_magnus = magnus_coefficient(velocity, spin)
    _, magnus = aerodynamic_force_components(velocity, spin)

    assert torch.isfinite(c_drag).all() and torch.isfinite(c_magnus).all()
    assert torch.all((c_drag > 0.3) & (c_drag < 0.7))
    assert magnus[2].norm() > magnus[1].norm() > magnus[0].norm()


def test_positive_topspin_accelerates_downward_for_both_directions():
    velocity = torch.tensor([[8.0, 0.0, 0.0], [-8.0, 0.0, 0.0]])
    command = torch.tensor([[300.0, 0.0, 0.0], [300.0, 0.0, 0.0]])
    spin = trajectory_spin_to_world(command, velocity)
    force = aerodynamic_force(velocity, spin)

    assert torch.all(force[:, 2] < 0.0)
    assert torch.allclose(world_spin_to_trajectory(spin, velocity), command, atol=1.0e-5)


def test_magnus_changes_flight_in_expected_direction():
    position = torch.tensor([[0.0, 0.0, 1.2]])
    velocity = torch.tensor([[8.0, 0.0, 2.0]])
    no_spin = torch.zeros_like(velocity)
    topspin = trajectory_spin_to_world(torch.tensor([[400.0, 0.0, 0.0]]), velocity)

    pos_no_spin, _, _ = integrate_flight(position, velocity, no_spin, duration=0.25)
    pos_topspin, _, _ = integrate_flight(position, velocity, topspin, duration=0.25)
    assert pos_topspin[0, 2] < pos_no_spin[0, 2]


def test_sidespin_mirrors_lateral_trajectory():
    position = torch.tensor([[0.0, 0.0, 1.2], [0.0, 0.0, 1.2]])
    velocity = torch.tensor([[8.0, 0.0, 1.0], [8.0, 0.0, 1.0]])
    command = torch.tensor([[0.0, 300.0, 0.0], [0.0, -300.0, 0.0]])
    spin = trajectory_spin_to_world(command, velocity)

    final_position, _, _ = integrate_flight(position, velocity, spin, duration=0.3)

    assert final_position[0, 1] > 0.0
    assert final_position[0, 1] == pytest.approx(-final_position[1, 1].item(), abs=1.0e-6)


def test_rk4_converges_at_simulator_timestep():
    position = torch.tensor([[0.0, 0.0, 1.3]])
    velocity = torch.tensor([[7.5, 0.4, 1.8]])
    spin = trajectory_spin_to_world(torch.tensor([[350.0, -180.0, 0.0]]), velocity)

    coarse_position, coarse_velocity, coarse_spin = integrate_flight(
        position, velocity, spin, duration=0.4, dt=0.002
    )
    fine_position, fine_velocity, fine_spin = integrate_flight(
        position, velocity, spin, duration=0.4, dt=0.001
    )

    assert torch.allclose(coarse_position, fine_position, atol=2.0e-4, rtol=0.0)
    assert torch.allclose(coarse_velocity, fine_velocity, atol=1.0e-3, rtol=0.0)
    assert torch.allclose(coarse_spin, fine_spin, atol=1.0e-6, rtol=0.0)


def test_fast_event_solver_matches_fine_integration_for_constant_coefficients():
    physics = replace(DEFAULT_BALL_PHYSICS, aerodynamic_model="constant")
    position = torch.tensor([[1.8, 0.1, 1.25], [1.8, -0.2, 1.18]])
    velocity = torch.tensor([[-6.0, 0.2, 1.4], [-7.0, -0.3, 0.8]])
    spin = torch.tensor([[0.0, -300.0, 100.0], [50.0, 250.0, 0.0]])

    fast = propagate_to_height(position, velocity, spin, 0.815, params=physics)
    reference = propagate_to_height_reference(
        position, velocity, spin, 0.815, dt=0.0005, max_time=1.2, params=physics
    )

    assert fast[-1].all() and reference[-1].all()
    assert torch.allclose(fast[0], reference[0], atol=2.0e-4, rtol=0.0)
    assert torch.allclose(fast[1], reference[1], atol=1.0e-3, rtol=0.0)
    assert torch.allclose(fast[2], reference[2], atol=2.0e-3, rtol=0.0)


def test_default_event_solvers_match_fine_state_dependent_aerodynamics():
    generator = torch.Generator().manual_seed(7)
    batch_size = 256
    position = torch.column_stack(
        (
            torch.full((batch_size,), 1.8),
            torch.empty(batch_size).uniform_(-0.35, 0.35, generator=generator),
            torch.empty(batch_size).uniform_(1.10, 1.32, generator=generator),
        )
    )
    velocity = torch.column_stack(
        (
            torch.empty(batch_size).uniform_(-7.0, -3.5, generator=generator),
            torch.empty(batch_size).uniform_(-0.8, 0.8, generator=generator),
            torch.empty(batch_size).uniform_(0.8, 2.5, generator=generator),
        )
    )
    command = torch.column_stack(
        (
            torch.linspace(-400.0, 400.0, batch_size),
            torch.empty(batch_size).uniform_(-250.0, 250.0, generator=generator),
            torch.zeros(batch_size),
        )
    )
    spin = trajectory_spin_to_world(command, velocity)

    fast_height = propagate_to_height(position, velocity, spin, 0.815, max_time=1.2)
    reference_height = propagate_to_height_reference(
        position, velocity, spin, 0.815, dt=0.0005, max_time=1.2
    )
    fast_x = propagate_to_x(position, velocity, spin, 0.0, max_time=1.2)
    reference_x = propagate_to_x_reference(
        position, velocity, spin, 0.0, dt=0.0005, max_time=1.2
    )

    assert fast_height[-1].all() and reference_height[-1].all()
    assert fast_x[-1].all() and reference_x[-1].all()
    height_error = torch.linalg.vector_norm(fast_height[1] - reference_height[1], dim=-1)
    x_error = torch.linalg.vector_norm(fast_x[1] - reference_x[1], dim=-1)
    assert torch.quantile(height_error, 0.95) < 1.5e-3
    assert torch.quantile(x_error, 0.95) < 5.0e-4

    # Numerical event convergence must not reject one spin direction more
    # often than the other.
    backspin = command[:, 0] < -100.0
    topspin = command[:, 0] > 100.0
    assert fast_height[-1][backspin].all()
    assert fast_height[-1][topspin].all()


def test_table_impact_couples_translation_and_spin():
    velocity = torch.tensor([[5.0, 0.0, -2.0]])
    spin = torch.zeros_like(velocity)
    out_velocity, out_spin, info = table_impact(velocity, spin)

    assert info["incoming"].item()
    assert out_velocity[0, 2] > 0.0
    assert out_velocity[0, 0] < velocity[0, 0]
    assert out_spin[0, 1] > 0.0


def test_table_does_not_bounce_ball_moving_away():
    velocity = torch.tensor([[5.0, 0.0, 2.0]])
    spin = torch.tensor([[0.0, 50.0, 0.0]])
    out_velocity, out_spin, info = table_impact(velocity, spin)

    assert not info["incoming"].item()
    assert torch.equal(out_velocity, velocity)
    assert torch.equal(out_spin, spin)


def test_brushing_racket_generates_spin():
    ball_velocity = torch.tensor([[4.0, 0.0, 0.0]])
    ball_spin = torch.zeros_like(ball_velocity)
    racket_velocity = torch.tensor([[-2.0, 0.0, 3.0]])
    racket_spin = torch.zeros_like(ball_velocity)
    normal = torch.tensor([[-1.0, 0.0, 0.0]])

    out_velocity, out_spin, info = racket_impact(
        ball_velocity,
        ball_spin,
        racket_velocity,
        racket_spin,
        normal,
    )
    assert info["incoming"].item()
    assert torch.linalg.vector_norm(out_spin) > 1.0
    assert torch.isfinite(out_velocity).all()
    assert torch.isfinite(out_spin).all()


def test_racket_impulse_respects_coulomb_cone_for_grazing_contact():
    ball_velocity = torch.tensor([[-0.02, 3.0, 0.0]])
    ball_spin = torch.zeros_like(ball_velocity)
    normal = torch.tensor([[1.0, 0.0, 0.0]])
    out_velocity, out_spin, info = racket_impact(
        ball_velocity,
        ball_spin,
        torch.zeros_like(ball_velocity),
        torch.zeros_like(ball_velocity),
        normal,
    )

    delta = out_velocity - ball_velocity
    normal_impulse = torch.abs(torch.sum(delta * info["normal"], dim=-1))
    tangent_impulse = torch.linalg.vector_norm(
        delta - torch.sum(delta * info["normal"], dim=-1, keepdim=True) * info["normal"],
        dim=-1,
    )
    assert info["friction_limited"].item()
    assert tangent_impulse <= DEFAULT_BALL_PHYSICS.racket_friction * normal_impulse + 1.0e-6
    assert torch.linalg.vector_norm(out_spin) < 5.0


@pytest.mark.parametrize(
    ("velocity", "expected_sliding"),
    [([0.05, 0.0, -3.0], False), ([8.0, 0.0, -0.2], True)],
)
def test_table_contact_selects_sticking_or_sliding_regime(velocity, expected_sliding):
    _, _, info = table_impact(torch.tensor([velocity]), torch.zeros(1, 3))
    assert info["sliding"].item() is expected_sliding
