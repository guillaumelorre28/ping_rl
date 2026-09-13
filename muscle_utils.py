from typing import Callable

import mujoco_warp as mjw
import torch
import warp as wp
from mujoco_warp._src.types import vec10
from mujoco_warp._src.util_misc import muscle_bias, muscle_gain


@wp.kernel
def muscle_bias_kernel(
    lengths: wp.array(dtype=float),
    lengthranges: wp.array(dtype=wp.vec2),
    acc0s: wp.array(dtype=float),
    prms: wp.array(dtype=vec10),
    out: wp.array(dtype=float),
):
    tid = wp.tid()
    out[tid] = muscle_bias(lengths[tid], lengthranges[tid], acc0s[tid], prms[tid])


@wp.kernel
def muscle_gain_kernel(
    lengths: wp.array(dtype=float),
    velocities: wp.array(dtype=float),
    lengthranges: wp.array(dtype=wp.vec2),
    acc0s: wp.array(dtype=float),
    prms: wp.array(dtype=vec10),
    out: wp.array(dtype=float),
):
    tid = wp.tid()
    out[tid] = muscle_gain(lengths[tid], velocities[tid], lengthranges[tid], acc0s[tid], prms[tid])


def get_target_actuator_length(
    model: mjw.Model,
    fk_data: mjw.Data,
    qpos: torch.Tensor,
    equalities: list[tuple[int, int, list[float]]],
    action_to_qpos: torch.Tensor,
    fk_forward: Callable,
):
    fk_data.qpos[:, action_to_qpos] = qpos

    for id1, id2, eq_data in equalities:
        qpos_id1 = model.jnt_qposadr[id1]
        qpos_id2 = model.jnt_qposadr[id2]

        if id2 == -1:
            fk_data.qpos[:, qpos_id1] = model.qpos0[:, qpos_id1] + eq_data[0]
        else:
            fk_data.qpos[:, qpos_id1] = torch.clamp(
                model.qpos0[:, qpos_id1]
                + eq_data[0]
                + eq_data[1] * (fk_data.qpos[:, qpos_id2] - model.qpos0[:, qpos_id2])
                + eq_data[2] * (fk_data.qpos[:, qpos_id2] - model.qpos0[:, qpos_id2]) ** 2
                + eq_data[3] * (fk_data.qpos[:, qpos_id2] - model.qpos0[:, qpos_id2]) ** 3
                + eq_data[4] * (fk_data.qpos[:, qpos_id2] - model.qpos0[:, qpos_id2]) ** 4,
                model.jnt_range[:, id1, 0],
                model.jnt_range[:, id1, 1],
            )

    fk_forward()
    return fk_data.actuator_length


def align_model_field(field: torch.Tensor, num_envs: int) -> torch.Tensor:
    """Broadcast a model field over the environments.

    MuJoCo Warp gives every model field a leading world dimension: 1 when the
    value is shared by all environments, ``num_envs`` once the field has been
    expanded (``Simulation.expand_model_fields``). Reading such a field with
    the actuator mask alone silently indexes that leading axis instead of the
    actuator axis, and deriving a batch size from it collapses it to 1 -- which
    makes the Warp launches below cover a fraction of the muscles and quietly
    reuse environment 0's coefficients everywhere. Going through this helper
    keeps the two cases apart and fails loudly on anything else.
    """

    if field.shape[0] == num_envs:
        return field
    if field.shape[0] == 1:
        return field.expand(num_envs, *field.shape[1:])
    raise ValueError(
        f"model field has leading dimension {field.shape[0]}, expected 1 or {num_envs}"
    )


def target_length_to_activations(
    model: mjw.Model,
    data: mjw.Data,
    target_length: torch.Tensor,
    kp_scale: torch.Tensor | float,
    kd_scale: torch.Tensor | float,
):

    device = target_length.device
    wp_device = wp.get_device(str(device))

    # `actuator_dyntype` carries no world dimension: the mask is a plain
    # per-actuator vector, applied to the actuator axis of everything below.
    muscle_indices = model.actuator_dyntype == 4  # mujoco.mjtDyn.mjDYN_MUSCLE

    length = data.actuator_length[:, muscle_indices]
    velocity = data.actuator_velocity[:, muscle_indices]
    # The data tensors are the only ones genuinely per-environment, so they --
    # not the model fields -- define the batch size.
    num_envs = length.shape[0]

    lengthrange = align_model_field(model.actuator_lengthrange, num_envs)[:, muscle_indices]
    biasprm = align_model_field(model.actuator_biasprm, num_envs)[:, muscle_indices]
    gainprm = align_model_field(model.actuator_gainprm, num_envs)[:, muscle_indices]
    acc0 = align_model_field(model.actuator_acc0, num_envs)[:, muscle_indices]
    peak_force = biasprm[..., 2]

    force = (
        (kp_scale * (target_length[:, muscle_indices] - length) - kd_scale * kp_scale * velocity)
        * peak_force
        / (lengthrange[..., 1] - lengthrange[..., 0])
    )
    clipped_force = torch.clamp(force, -peak_force, torch.zeros_like(peak_force))

    prmb = biasprm[..., :10]
    prmg = gainprm[..., :10]

    num_muscles = length.shape[1]
    total = num_envs * num_muscles
    # `align_model_field` returns expanded views for shared fields; Warp reads
    # raw buffers, so materialise them before handing them over.
    flat_lengthrange = lengthrange.reshape(-1, 2).contiguous()
    flat_acc0 = acc0.reshape(-1).contiguous()
    flat_length = length.reshape(-1).contiguous()
    flat_velocity = velocity.reshape(-1).contiguous()

    with wp.ScopedDevice(wp_device):
        bias = wp.zeros(total, dtype=float, device=wp_device)
        wp.launch(
            muscle_bias_kernel,
            dim=total,
            inputs=[
                flat_length,
                flat_lengthrange,
                flat_acc0,
                prmb.reshape(-1, 10).contiguous(),
                bias,
            ],
            device=bias.device,
        )

        gain = wp.zeros(total, dtype=float, device=wp_device)
        wp.launch(
            muscle_gain_kernel,
            dim=total,
            inputs=[
                flat_length,
                flat_velocity,
                flat_lengthrange,
                flat_acc0,
                prmg.reshape(-1, 10).contiguous(),
                gain,
            ],
            device=gain.device,
        )

        bias = wp.to_torch(bias).reshape(num_envs, num_muscles)
        gain = wp.to_torch(gain).reshape(num_envs, num_muscles)
        gain = torch.clamp(gain, max=-1)
        activations = torch.clamp((clipped_force - bias) / gain, 0, 1)
    return activations, bias, gain, clipped_force


def calculate_vae_muscle_act(
    model: mjw.Model,
    data: mjw.Data,
    target_length: torch.Tensor,
    kp: float = 10.0,
    kd: float = 1.0,
) -> torch.Tensor:
    """
    Compute muscle activations for MuscleVAE control.

    Uses a PD controller to compute the desired contraction force, then solves
    inverse dynamics via MuJoCo's FLV curve parameters to convert target muscle
    lengths into activation signals.

    Pipeline:
        1. PD force:      f_pd = kp * (l_target - l_current) - kd * v_current
        2. Normalize:     f_normalized = f_pd * F0 / (l_max - l_min)
        3. Clip:          f_clipped = clamp(f_normalized, -F0, 0)  # muscles can only pull
        4. Inverse dyn:   activation = (f_clipped - bias) / gain

    Args:
        model: MuJoCo Warp model containing muscle parameters.
        data: MuJoCo Warp data with current muscle state.
        target_length: Target muscle length tensor [batch_size, num_actuators].
        kp: Proportional gain for PD controller (default: 10.0).
        kd: Derivative gain for PD controller (default: 1.0).

    Returns:
        activations: Muscle activation tensor [batch_size, num_muscles] in [0, 1].
    """
    device = target_length.device
    wp_device = wp.get_device(str(device))

    # dyntype == 4 corresponds to mjDYN_MUSCLE
    muscle_indices = model.actuator_dyntype == 4

    # Step 1: extract current muscle state. Data tensors are the only genuinely
    # per-environment ones, so they define the batch size; model fields are
    # broadcast through `align_model_field` (see there for why).
    length = data.actuator_length[:, muscle_indices]          # [batch, num_muscles]
    velocity = data.actuator_velocity[:, muscle_indices]      # [batch, num_muscles]
    num_envs = length.shape[0]
    lengthrange = align_model_field(model.actuator_lengthrange, num_envs)[:, muscle_indices]
    biasprm = align_model_field(model.actuator_biasprm, num_envs)[:, muscle_indices]
    gainprm = align_model_field(model.actuator_gainprm, num_envs)[:, muscle_indices]
    acc0 = align_model_field(model.actuator_acc0, num_envs)[:, muscle_indices]
    F0 = biasprm[..., 2]                                      # peak isometric force

    # Step 2: PD force, normalized to physical units
    length_error = target_length[:, muscle_indices] - length
    f_pd = (kp * length_error - kd * velocity) * F0 / (lengthrange[..., 1] - lengthrange[..., 0])

    # Step 3: muscles can only pull (negative force), clamp to [-F0, 0]
    f_clipped = torch.clamp(f_pd, min=-F0, max=torch.zeros_like(F0))

    # Step 4: inverse dynamics via FLV curve (bias/gain computed by Warp kernels)
    prmb = biasprm[..., :10]                               # [batch, num_muscles, 10]
    prmg = gainprm[..., :10]                               # [batch, num_muscles, 10]

    batch_size = num_envs
    num_muscles = length.shape[1]
    # Expanded views are not contiguous; Warp reads raw buffers.
    flat_lengthrange = lengthrange.reshape(-1, 2).contiguous()
    flat_acc0 = acc0.reshape(-1).contiguous()
    flat_length = length.reshape(-1).contiguous()
    flat_velocity = velocity.reshape(-1).contiguous()

    with wp.ScopedDevice(wp_device):
        bias_flat = wp.zeros(batch_size * num_muscles, dtype=float, device=wp_device)
        wp.launch(
            muscle_bias_kernel,
            dim=batch_size * num_muscles,
            inputs=[
                flat_length,
                flat_lengthrange,
                flat_acc0,
                prmb.reshape(-1, 10).contiguous(),
                bias_flat,
            ],
            device=bias_flat.device,
        )

        gain_flat = wp.zeros(batch_size * num_muscles, dtype=float, device=wp_device)
        wp.launch(
            muscle_gain_kernel,
            dim=batch_size * num_muscles,
            inputs=[
                flat_length,
                flat_velocity,
                flat_lengthrange,
                flat_acc0,
                prmg.reshape(-1, 10).contiguous(),
                gain_flat,
            ],
            device=gain_flat.device,
        )

        bias = wp.to_torch(bias_flat).reshape(batch_size, num_muscles)
        gain = wp.to_torch(gain_flat).reshape(batch_size, num_muscles)

        gain = torch.clamp(gain, max=-1.0)  # gain must be negative for numerical stability
        activations = torch.clamp((f_clipped - bias) / gain, min=0.0, max=1.0)

    return activations
