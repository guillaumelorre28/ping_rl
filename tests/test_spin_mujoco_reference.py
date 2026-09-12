"""Native-contact characterization using the project's actual MuJoCo model."""

from pathlib import Path

import mujoco
import pytest
import torch
from ball_physics import DEFAULT_BALL_PHYSICS, racket_impact, table_impact

MODEL_PATH = Path(__file__).resolve().parents[1] / "tabletennis.xml"


def _has_target_contact(
    data: mujoco.MjData,
    ball_geom_id: int,
    target_geom_id: int,
) -> bool:
    target_pair = {ball_geom_id, target_geom_id}
    return any(
        {data.contact[index].geom1, data.contact[index].geom2} == target_pair
        for index in range(data.ncon)
    )


def _build_project_contact_scene(
    velocity: list[float],
    spin: list[float],
    *,
    contact: str,
    timestep: float,
) -> tuple[mujoco.MjModel, mujoco.MjData, int, int, int]:
    """Set up one isolated ball contact with the real project geometry/settings.

    Returned as a model/data pair so the same scene can be advanced by either
    engine: MuJoCo on CPU for characterization, MuJoCo-Warp for the engine the
    training environment actually runs.
    """

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.opt.timestep = timestep
    # Gravity is excluded over the short contact so the result represents only
    # the contact impulse, as does the instantaneous analytic map.
    model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)

    ball_body_id = model.body("pingpong").id
    model.body_inertia[ball_body_id, :] = DEFAULT_BALL_PHYSICS.shell_inertia
    mujoco.mj_setConst(model, data)

    ball_joint_id = model.joint("pingpong_freejoint").id
    ball_qpos_address = model.jnt_qposadr[ball_joint_id]
    ball_dof_address = model.jnt_dofadr[ball_joint_id]
    paddle_joint_id = model.joint("paddle_freejoint").id
    paddle_qpos_address = model.jnt_qposadr[paddle_joint_id]

    data.qpos[:] = model.key_qpos[0]
    data.qvel[:] = 0.0
    if contact == "table":
        # The own table surface is z=0.795 m; start 2 mm above first touch.
        data.qpos[ball_qpos_address : ball_qpos_address + 7] = (
            0.5,
            0.0,
            0.817,
            1.0,
            0.0,
            0.0,
            0.0,
        )
        target_geom_id = model.geom("coll_own_half").id
    elif contact == "racket":
        # Rotate the actual cylindrical pad so its local -z face points +x,
        # then launch the ball at that face with a tangential z component.
        data.qpos[paddle_qpos_address : paddle_qpos_address + 7] = (
            0.0,
            0.0,
            1.2,
            0.70710678,
            0.0,
            -0.70710678,
            0.0,
        )
        data.qpos[ball_qpos_address : ball_qpos_address + 7] = (
            -0.043,
            0.0,
            1.13,
            1.0,
            0.0,
            0.0,
            0.0,
        )
        target_geom_id = model.geom("pad").id
    else:
        raise ValueError(f"Unknown contact type: {contact}")

    data.qvel[ball_dof_address : ball_dof_address + 6] = velocity + spin
    mujoco.mj_forward(model, data)
    return model, data, ball_dof_address, model.geom("pingpong").id, target_geom_id


def _split_ball_twist(
    qvel: "object", ball_dof_address: int
) -> tuple[torch.Tensor, torch.Tensor]:
    linear = torch.tensor(
        qvel[ball_dof_address : ball_dof_address + 3].copy(), dtype=torch.float32
    )
    angular = torch.tensor(
        qvel[ball_dof_address + 3 : ball_dof_address + 6].copy(), dtype=torch.float32
    )
    return linear.unsqueeze(0), angular.unsqueeze(0)


def _project_model_native_impact(
    velocity: list[float],
    spin: list[float],
    *,
    contact: str,
    timestep: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance one isolated project contact with MuJoCo on CPU."""

    model, data, ball_dof_address, ball_geom_id, target_geom_id = (
        _build_project_contact_scene(velocity, spin, contact=contact, timestep=timestep)
    )
    made_contact = False
    for _ in range(200):
        mujoco.mj_step(model, data)
        active = _has_target_contact(data, ball_geom_id, target_geom_id)
        made_contact |= active
        if made_contact and not active:
            break

    assert made_contact
    return _split_ball_twist(data.qvel, ball_dof_address)


def _project_model_warp_impact(
    velocity: list[float],
    spin: list[float],
    *,
    contact: str,
    timestep: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance the same scene with MuJoCo-Warp, the engine training uses."""

    mujoco_warp = pytest.importorskip("mujoco_warp")

    model, data, ball_dof_address, ball_geom_id, target_geom_id = (
        _build_project_contact_scene(velocity, spin, contact=contact, timestep=timestep)
    )
    warp_model = mujoco_warp.put_model(model)
    warp_data = mujoco_warp.put_data(model, data, nworld=1, nconmax=2000, njmax=500)

    target_pair = {ball_geom_id, target_geom_id}
    made_contact = False
    for _ in range(200):
        mujoco_warp.step(warp_model, warp_data)
        geoms = warp_data.contact.geom.numpy()
        count = min(int(warp_data.nacon.numpy()[0]), geoms.shape[0])
        active = any(
            {int(geoms[index, 0]), int(geoms[index, 1])} == target_pair
            for index in range(count)
        )
        made_contact |= active
        if made_contact and not active:
            break

    assert made_contact
    return _split_ball_twist(warp_data.qvel.numpy()[0], ball_dof_address)


def test_project_table_contact_is_characterized_at_2ms():
    velocity = torch.tensor([[5.0, 0.0, -2.0]])
    spin = torch.tensor([[0.0, 150.0, 0.0]])
    native_velocity, native_spin = _project_model_native_impact(
        velocity[0].tolist(), spin[0].tolist(), contact="table", timestep=0.002
    )
    analytic_velocity, analytic_spin, _ = table_impact(velocity, spin)

    assert torch.allclose(
        native_velocity, torch.tensor([[4.9889, 0.0, 1.9512]]), atol=0.06, rtol=0.0
    )
    assert torch.allclose(
        native_spin, torch.tensor([[0.0, 150.7330, 0.0]]), atol=1.0, rtol=0.0
    )
    # Record the material difference behind analytic_contact_override rather
    # than hiding it behind a broad "agreement" tolerance.
    assert torch.linalg.vector_norm(native_velocity - analytic_velocity).item() > 0.7
    assert torch.linalg.vector_norm(native_spin - analytic_spin).item() > 50.0


def test_project_racket_contact_is_characterized_at_2ms():
    velocity = torch.tensor([[5.0, 0.0, 2.0]])
    spin = torch.zeros_like(velocity)
    zeros = torch.zeros_like(velocity)
    normal = torch.tensor([[-1.0, 0.0, 0.0]])
    native_velocity, native_spin = _project_model_native_impact(
        velocity[0].tolist(), spin[0].tolist(), contact="racket", timestep=0.002
    )
    analytic_velocity, analytic_spin, _ = racket_impact(
        velocity, spin, zeros, zeros, normal
    )

    assert torch.allclose(
        native_velocity, torch.tensor([[-4.7283, 0.0, 1.9562]]), atol=0.06, rtol=0.0
    )
    assert torch.allclose(
        native_spin, torch.tensor([[0.0, 1.1476, 0.0]]), atol=1.0, rtol=0.0
    )
    assert torch.linalg.vector_norm(native_velocity - analytic_velocity).item() > 1.0
    assert torch.linalg.vector_norm(native_spin - analytic_spin).item() > 80.0


@pytest.mark.parametrize(
    ("contact", "velocity", "spin", "velocity_tolerance"),
    [
        ("table", [5.0, 0.0, -2.0], [0.0, 150.0, 0.0], 0.06),
        ("racket", [5.0, 0.0, 2.0], [0.0, 0.0, 0.0], 0.02),
    ],
)
def test_project_native_contact_converges_from_2ms_to_half_ms(
    contact: str,
    velocity: list[float],
    spin: list[float],
    velocity_tolerance: float,
):
    coarse_velocity, coarse_spin = _project_model_native_impact(
        velocity, spin, contact=contact, timestep=0.002
    )
    middle_velocity, middle_spin = _project_model_native_impact(
        velocity, spin, contact=contact, timestep=0.001
    )
    fine_velocity, fine_spin = _project_model_native_impact(
        velocity, spin, contact=contact, timestep=0.0005
    )

    assert torch.allclose(
        coarse_velocity, fine_velocity, atol=velocity_tolerance, rtol=0.0
    )
    assert torch.allclose(
        middle_velocity, fine_velocity, atol=velocity_tolerance / 2.0, rtol=0.0
    )
    assert torch.allclose(coarse_spin, fine_spin, atol=1.0, rtol=0.0)
    assert torch.allclose(middle_spin, fine_spin, atol=0.5, rtol=0.0)


@pytest.mark.slow
@pytest.mark.parametrize("timestep", [0.002, 0.001, 0.0005])
@pytest.mark.parametrize(
    ("contact", "velocity", "spin"),
    [
        ("table", [5.0, 0.0, -2.0], [0.0, 150.0, 0.0]),
        ("racket", [5.0, 0.0, 2.0], [0.0, 0.0, 0.0]),
    ],
)
def test_warp_engine_reproduces_cpu_native_contact(
    contact: str,
    velocity: list[float],
    spin: list[float],
    timestep: float,
):
    """Pin the CPU characterization to the engine the training loop runs.

    Every other test in this file advances the scene with ``mujoco.mj_step``,
    while the environment steps MuJoCo-Warp.  Without this check the recorded
    native behaviour -- and therefore the meaning of the
    ``analytic_contact_override: false`` ablation -- would only be established
    for CPU MuJoCo.
    """

    cpu_velocity, cpu_spin = _project_model_native_impact(
        velocity, spin, contact=contact, timestep=timestep
    )
    warp_velocity, warp_spin = _project_model_warp_impact(
        velocity, spin, contact=contact, timestep=timestep
    )

    assert torch.allclose(cpu_velocity, warp_velocity, atol=2.0e-3, rtol=0.0)
    assert torch.allclose(cpu_spin, warp_spin, atol=5.0e-2, rtol=0.0)
