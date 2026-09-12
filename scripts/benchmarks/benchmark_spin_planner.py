"""Benchmark the fixed-cost spin planner and its event-solver accuracy."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ball_physics import (
    propagate_to_height,
    propagate_to_height_reference,
    trajectory_spin_to_world,
    world_spin_to_trajectory,
)
from planner import IncomingHitPrediction, plan_spin_return


def make_batch(batch_size: int, seed: int) -> tuple[IncomingHitPrediction, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    position = torch.column_stack(
        (
            torch.full((batch_size,), 1.8),
            torch.empty(batch_size).uniform_(-0.35, 0.35, generator=generator),
            torch.empty(batch_size).uniform_(1.12, 1.32, generator=generator),
        )
    )
    velocity = torch.column_stack(
        (
            torch.empty(batch_size).uniform_(5.5, 8.0, generator=generator),
            torch.empty(batch_size).uniform_(-0.5, 0.5, generator=generator),
            torch.empty(batch_size).uniform_(0.2, 2.0, generator=generator),
        )
    )
    incoming_command = torch.column_stack(
        (
            torch.empty(batch_size).uniform_(-300.0, 300.0, generator=generator),
            torch.empty(batch_size).uniform_(-200.0, 200.0, generator=generator),
            torch.zeros(batch_size),
        )
    )
    incoming = IncomingHitPrediction(
        position=position,
        velocity=velocity,
        spin=trajectory_spin_to_world(incoming_command, velocity),
        time=torch.zeros(batch_size),
        valid=torch.ones(batch_size, dtype=torch.bool),
    )
    landing = torch.column_stack(
        (
            torch.empty(batch_size).uniform_(-1.3, -0.55, generator=generator),
            torch.empty(batch_size).uniform_(-0.35, 0.35, generator=generator),
            torch.full((batch_size,), 0.815),
        )
    )
    # Sweep the commanded topspin instead of sampling it, so heavy backspin --
    # the longest flights and the hardest case for the event solver -- is
    # always represented at a fixed proportion of the batch.
    command = torch.column_stack(
        (
            torch.linspace(-400.0, 400.0, batch_size),
            torch.empty(batch_size).uniform_(-250.0, 250.0, generator=generator),
            torch.zeros(batch_size),
        )
    )
    return incoming, landing, command


def _summarize(label: str, values: torch.Tensor, scale: float, unit: str) -> None:
    if values.numel() == 0:
        print(f"  {label:<26} (empty)")
        return
    scaled = values * scale
    print(
        f"  {label:<26} median={scaled.median():8.2f}  "
        f"p95={torch.quantile(scaled, 0.95):8.2f}  max={scaled.max():8.2f} {unit}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    incoming, landing, command = make_batch(args.batch_size, seed=19)
    plan_spin_return(incoming, landing, command)
    durations = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        plan = plan_spin_return(incoming, landing, command)
        durations.append(1.0e3 * (time.perf_counter() - start))

    # Characterise the event solver on the states the planner actually emits,
    # and score the plan against fine integration rather than against its own
    # predictor -- self-consistency passes even when the fast solver drifts.
    reference = propagate_to_height_reference(
        incoming.position,
        plan.predicted_out_velocity,
        plan.predicted_out_spin,
        0.815,
        dt=0.0005,
        max_time=1.5,
    )
    fast = propagate_to_height(
        incoming.position,
        plan.predicted_out_velocity,
        plan.predicted_out_spin,
        0.815,
        max_time=1.5,
    )
    common = reference[-1] & fast[-1]
    event_error = torch.linalg.vector_norm(reference[1][common] - fast[1][common], dim=-1)

    evaluated = plan.valid & reference[-1]
    true_landing_error = torch.linalg.vector_norm(
        reference[1][evaluated][:, :2] - landing[evaluated][:, :2], dim=-1
    )
    self_landing_error = torch.linalg.vector_norm(
        plan.predicted_landing_position[:, :2] - landing[:, :2], dim=-1
    )

    achieved = world_spin_to_trajectory(plan.predicted_out_spin, plan.predicted_out_velocity)
    spin_error = (achieved[:, :2] - command[:, :2]).abs().amax(dim=-1)
    commanded_magnitude = torch.linalg.vector_norm(command[:, :2], dim=-1)

    print(f"batch={args.batch_size}  median={statistics.median(durations):.2f} ms")
    print(f"valid plans={plan.valid.float().mean().item():.3%}")
    print(f"event solver rejects={((~fast[-1]) & reference[-1]).float().mean().item():.3%}")
    print("accuracy:")
    _summarize("landing (fine integration)", true_landing_error, 1.0e3, "mm")
    _summarize("landing (self-reported)", self_landing_error, 1.0e3, "mm")
    _summarize("event solver vs fine", event_error, 1.0e3, "mm")
    print("spin tracking by commanded magnitude:")
    for low, high in ((0.0, 150.0), (150.0, 250.0), (250.0, 325.0), (325.0, 500.0)):
        band = (commanded_magnitude >= low) & (commanded_magnitude < high)
        _summarize(f"|command| in [{low:.0f},{high:.0f})", spin_error[band], 1.0, "rad/s")
    print("spin tracking by direction (|command| <= 250):")
    reachable = commanded_magnitude <= 250.0
    for label, mask in (
        ("backspin", reachable & (command[:, 0] < -100.0)),
        ("topspin", reachable & (command[:, 0] > 100.0)),
    ):
        _summarize(label, spin_error[mask], 1.0, "rad/s")


if __name__ == "__main__":
    main()
