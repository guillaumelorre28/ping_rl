"""Spin-aware table-tennis ball physics shared by simulation and planning.

The models in this module deliberately stay independent from MuJoCo.  This
makes them usable as a numerical oracle in tests and prevents the planner from
quietly using different ball dynamics than the RL environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch


@dataclass(frozen=True)
class BallPhysicsParams:
    """Physical and identified parameters for a 40 mm table-tennis ball."""

    mass: float = 2.7e-3
    radius: float = 2.0e-2
    air_density: float = 1.204
    # ``conti2026`` uses the speed/spin-dependent fits reported in
    # arXiv:2606.28805.  ``constant`` is kept as a cheap, explicit ablation.
    aerodynamic_model: str = "conti2026"
    drag_enabled: bool = True
    magnus_enabled: bool = True
    # Constant-model values reproduce TT4D's k_d=3.8e-4 and k_m=3.0e-6
    # after conversion to the force convention used here.
    drag_coefficient: float = 0.5023162656
    magnus_coefficient: float = 0.0743560262
    spin_decay: float = 0.0
    gravity: float = 9.81

    table_friction: float = 0.25
    table_restitution_base: float = 0.98
    table_restitution_velocity_slope: float = 0.02
    table_restitution_min: float = 0.60
    table_restitution_max: float = 0.99

    racket_normal_restitution_base: float = 0.878
    racket_normal_restitution_velocity_slope: float = -0.020
    racket_normal_restitution_min: float = 0.20
    racket_normal_restitution_max: float = 0.95
    racket_tangential_restitution_base: float = 0.819
    racket_tangential_restitution_velocity_slope: float = -0.010
    racket_tangential_restitution_min: float = -0.20
    racket_tangential_restitution_max: float = 0.95
    racket_normal_spin_retention: float = 0.805
    racket_friction: float = 0.80

    @property
    def area(self) -> float:
        return math.pi * self.radius**2

    @property
    def volume(self) -> float:
        return 4.0 * math.pi * self.radius**3 / 3.0

    @property
    def shell_inertia(self) -> float:
        return 2.0 * self.mass * self.radius**2 / 3.0

    @property
    def inertia_ratio(self) -> float:
        """Return m r^2 / I; equal to 3/2 for a thin spherical shell."""

        return self.mass * self.radius**2 / self.shell_inertia


DEFAULT_BALL_PHYSICS = BallPhysicsParams()


@lru_cache(maxsize=16)
def _conti_aerodynamic_tables(device: str, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    """Return cached coefficient tables from Conti et al. (2026)."""

    kwargs = {"device": torch.device(device), "dtype": dtype}
    drag_speeds = torch.tensor((2.5, 7.5, 12.5, 17.5), **kwargs)
    drag_spin_ratios = torch.tensor(
        (
            (0.0, 0.3, 0.7, 0.95, 1.5, 2.0),
            (0.0, 0.4, 0.75, 1.1, 1.3, 2.0),
            (0.0, 0.4, 0.62, 0.95, 1.3, 2.0),
            (0.0, 0.4, 0.5, 0.84, 1.2, 2.0),
        ),
        **kwargs,
    )
    drag_coefficients = torch.tensor(
        (
            (0.55, 0.55, 0.55, 0.55, 0.55, 0.55),
            (0.49, 0.49, 0.55, 0.48, 0.53, 0.53),
            (0.47, 0.47, 0.53, 0.41, 0.48, 0.48),
            (0.47, 0.47, 0.51, 0.37, 0.45, 0.45),
        ),
        **kwargs,
    )
    magnus_speeds = torch.tensor((2.0, 3.5, 7.5, 10.5, 13.5, 17.0), **kwargs)
    magnus_linear = torch.tensor(
        (
            (0.0, 0.08, 150.0),
            (-1.1e-3, 0.31, 200.0),
            (-8.0e-4, 0.37, 350.0),
            (-6.58e-4, 0.375, 440.0),
            (-5.6e-4, 0.383, 550.0),
            (-4.48e-4, 0.371, 650.0),
        ),
        **kwargs,
    )
    magnus_quadratic = torch.tensor(
        (
            (-1.852e-7, -1.296e-4, 0.0983),
            (-1.667e-7, -3.333e-5, 0.1),
            (-2.0e-7, 1.7e-4, 0.0587),
            (-2.604e-7, 3.646e-4, -0.0225),
            (-3.571e-7, 5.357e-4, -0.0893),
            (-1.0e-7, 2.3e-4, -0.0375),
        ),
        **kwargs,
    )
    return (
        drag_speeds,
        drag_spin_ratios,
        drag_coefficients,
        magnus_speeds,
        magnus_linear,
        magnus_quadratic,
    )


def _normalize(vector: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(eps)


def trajectory_spin_basis(velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return lateral, up and forward axes associated with a trajectory.

    A positive rotation around the lateral axis produces topspin: with the
    Magnus convention used below, its acceleration points downwards.
    """

    forward = velocity.clone()
    forward[..., 2] = 0.0
    fallback = torch.zeros_like(forward)
    fallback[..., 0] = 1.0
    forward_norm = torch.linalg.vector_norm(forward, dim=-1, keepdim=True)
    forward = torch.where(forward_norm > 1.0e-8, forward / forward_norm.clamp_min(1.0e-8), fallback)

    up = torch.zeros_like(forward)
    up[..., 2] = 1.0
    lateral = _normalize(torch.cross(up, forward, dim=-1))
    return lateral, up, forward


def trajectory_spin_to_world(spin_command: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    """Convert ``(topspin, sidespin, corkscrew)`` from trajectory to world axes."""

    lateral, up, forward = trajectory_spin_basis(velocity)
    return (
        spin_command[..., 0:1] * lateral
        + spin_command[..., 1:2] * up
        + spin_command[..., 2:3] * forward
    )


def world_spin_to_trajectory(spin: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    """Convert world angular velocity to ``(topspin, sidespin, corkscrew)``."""

    lateral, up, forward = trajectory_spin_basis(velocity)
    return torch.stack(
        (
            torch.sum(spin * lateral, dim=-1),
            torch.sum(spin * up, dim=-1),
            torch.sum(spin * forward, dim=-1),
        ),
        dim=-1,
    )


def _piecewise_linear_rows(query: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Evaluate one piecewise-linear function per row of ``x``/``y``.

    ``x`` and ``y`` are ``(rows, knots)`` with ascending knots along each row.
    The result appends a trailing ``rows`` dimension to ``query``.  Searching
    every row in a single batched ``searchsorted`` keeps the coefficient tables
    off the per-substep critical path: this runs once instead of once per row.
    """

    rows, knots = x.shape
    flat = query.reshape(1, -1).expand(rows, -1)
    clipped = flat.clamp(x[:, :1], x[:, -1:]).contiguous()
    upper = torch.searchsorted(x.contiguous(), clipped, right=True).clamp(1, knots - 1)
    lower = upper - 1
    x0, x1 = torch.gather(x, 1, lower), torch.gather(x, 1, upper)
    y0, y1 = torch.gather(y, 1, lower), torch.gather(y, 1, upper)
    weight = (clipped - x0) / (x1 - x0).clamp_min(torch.finfo(query.dtype).eps)
    values = y0 + weight * (y1 - y0)
    return values.transpose(0, 1).reshape(*query.shape, rows)


def _interpolate_reference_speeds(
    speed: torch.Tensor,
    reference_speeds: torch.Tensor,
    row_values: torch.Tensor,
    *,
    extrapolate_high: bool = False,
) -> torch.Tensor:
    """Interpolate values fitted independently at several reference speeds."""

    query = speed.clamp_min(reference_speeds[0])
    if not extrapolate_high:
        query = query.clamp_max(reference_speeds[-1])
    upper = torch.searchsorted(reference_speeds, query.contiguous(), right=True).clamp(
        1, reference_speeds.shape[0] - 1
    )
    lower = upper - 1
    lower_value = torch.gather(row_values, -1, lower.unsqueeze(-1)).squeeze(-1)
    upper_value = torch.gather(row_values, -1, upper.unsqueeze(-1)).squeeze(-1)
    x0, x1 = reference_speeds[lower], reference_speeds[upper]
    weight = (query - x0) / (x1 - x0).clamp_min(torch.finfo(speed.dtype).eps)
    return lower_value + weight * (upper_value - lower_value)


def drag_coefficient(
    velocity: torch.Tensor,
    spin: torch.Tensor,
    params: BallPhysicsParams = DEFAULT_BALL_PHYSICS,
) -> torch.Tensor:
    """Return the drag coefficient for each ball state."""

    if not params.drag_enabled:
        return torch.zeros(velocity.shape[:-1], device=velocity.device, dtype=velocity.dtype)
    if params.aerodynamic_model == "constant":
        return torch.full(velocity.shape[:-1], params.drag_coefficient, device=velocity.device, dtype=velocity.dtype)
    if params.aerodynamic_model != "conti2026":
        raise ValueError(f"Unknown aerodynamic model: {params.aerodynamic_model}")

    speed = torch.linalg.vector_norm(velocity, dim=-1)
    spin_speed = torch.linalg.vector_norm(spin, dim=-1)
    spin_ratio = params.radius * spin_speed / speed.clamp_min(1.0e-6)
    tables = _conti_aerodynamic_tables(str(velocity.device), velocity.dtype)
    reference_speeds, ratio_knots, coefficient_knots = tables[:3]
    values_at_reference_speeds = _piecewise_linear_rows(
        spin_ratio, ratio_knots, coefficient_knots
    )
    return _interpolate_reference_speeds(
        speed,
        reference_speeds,
        values_at_reference_speeds,
        extrapolate_high=True,
    ).clamp_min(0.0)


def magnus_coefficient(
    velocity: torch.Tensor,
    spin: torch.Tensor,
    params: BallPhysicsParams = DEFAULT_BALL_PHYSICS,
) -> torch.Tensor:
    """Return a Magnus coefficient without the old quadratic-spin artefact.

    The default is the speed/spin-dependent fit of Conti et al. (2026).  The
    constant alternative gives the common ``rho * V * C_M * (omega x v)``
    model and is useful for controlled ablations.
    """

    if not params.magnus_enabled:
        return torch.zeros(velocity.shape[:-1], device=velocity.device, dtype=velocity.dtype)
    if params.aerodynamic_model == "constant":
        return torch.full(velocity.shape[:-1], params.magnus_coefficient, device=velocity.device, dtype=velocity.dtype)
    if params.aerodynamic_model != "conti2026":
        raise ValueError(f"Unknown aerodynamic model: {params.aerodynamic_model}")

    speed = torch.linalg.vector_norm(velocity, dim=-1)
    spin_speed = torch.linalg.vector_norm(spin, dim=-1)
    tables = _conti_aerodynamic_tables(str(velocity.device), velocity.dtype)
    reference_speeds, linear, quadratic = tables[3:]
    omega = spin_speed.unsqueeze(-1)
    linear_values = linear[:, 0] * omega + linear[:, 1]
    quadratic_values = quadratic[:, 0] * omega**2 + quadratic[:, 1] * omega + quadratic[:, 2]
    values_at_reference_speeds = torch.where(
        omega <= linear[:, 2], linear_values, quadratic_values
    ).clamp_min(0.0)
    return _interpolate_reference_speeds(
        speed, reference_speeds, values_at_reference_speeds, extrapolate_high=False
    )


def aerodynamic_force_components(
    velocity: torch.Tensor,
    spin: torch.Tensor,
    params: BallPhysicsParams = DEFAULT_BALL_PHYSICS,
    *,
    coefficients: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return separate drag and Magnus forces in world coordinates."""

    speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
    if coefficients is None:
        c_drag = drag_coefficient(velocity, spin, params)
        c_magnus = magnus_coefficient(velocity, spin, params)
    else:
        c_drag, c_magnus = coefficients
    c_drag = c_drag.unsqueeze(-1)
    drag = -0.5 * c_drag * params.air_density * params.area * speed * velocity
    c_magnus = c_magnus.unsqueeze(-1)
    magnus = (
        c_magnus
        * params.air_density
        * params.volume
        * torch.cross(spin, velocity, dim=-1)
    )
    return drag, magnus


def aerodynamic_force(
    velocity: torch.Tensor,
    spin: torch.Tensor,
    params: BallPhysicsParams = DEFAULT_BALL_PHYSICS,
) -> torch.Tensor:
    """Return drag plus Magnus force, excluding gravity, in world coordinates."""

    drag, magnus = aerodynamic_force_components(velocity, spin, params)
    return drag + magnus


def flight_acceleration(
    velocity: torch.Tensor,
    spin: torch.Tensor,
    params: BallPhysicsParams = DEFAULT_BALL_PHYSICS,
    *,
    coefficients: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Return total free-flight acceleration including gravity."""

    drag, magnus = aerodynamic_force_components(
        velocity, spin, params, coefficients=coefficients
    )
    acceleration = (drag + magnus) / params.mass
    gravity = torch.zeros_like(acceleration)
    gravity[..., 2] = -params.gravity
    return acceleration + gravity


def rk4_step(
    position: torch.Tensor,
    velocity: torch.Tensor,
    spin: torch.Tensor,
    dt: float | torch.Tensor,
    params: BallPhysicsParams = DEFAULT_BALL_PHYSICS,
    *,
    coefficients: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance free-flight ball state by one fixed RK4 step."""

    step = torch.as_tensor(dt, dtype=velocity.dtype, device=velocity.device)
    while step.ndim < velocity.ndim:
        step = step.unsqueeze(-1)

    def derivative(v: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            v,
            flight_acceleration(v, w, params, coefficients=coefficients),
            -params.spin_decay * w,
        )

    k1_p, k1_v, k1_w = derivative(velocity, spin)
    k2_p, k2_v, k2_w = derivative(velocity + 0.5 * step * k1_v, spin + 0.5 * step * k1_w)
    k3_p, k3_v, k3_w = derivative(velocity + 0.5 * step * k2_v, spin + 0.5 * step * k2_w)
    k4_p, k4_v, k4_w = derivative(velocity + step * k3_v, spin + step * k3_w)

    next_position = position + step * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p) / 6.0
    next_velocity = velocity + step * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0
    next_spin = spin + step * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w) / 6.0
    return next_position, next_velocity, next_spin


def integrate_flight(
    position: torch.Tensor,
    velocity: torch.Tensor,
    spin: torch.Tensor,
    duration: float,
    dt: float = 0.002,
    params: BallPhysicsParams = DEFAULT_BALL_PHYSICS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrate a free flight for a fixed duration."""

    steps = max(1, math.ceil(duration / dt))
    step_dt = duration / steps
    pos, vel, angvel = position, velocity, spin
    for _ in range(steps):
        pos, vel, angvel = _rk4(pos, vel, angvel, step_dt, params)
    return pos, vel, angvel



# --- compilation du pas d'intégration ------------------------------------
#
# Profilé sur RTX 3090 le 15 septembre 2026 : le planificateur représentait
# 88 % du temps d'un pas d'entraînement et la physique MuJoCo Warp 2,8 %. La
# cause n'était pas le calcul mais la latence : 368 412 lancements de noyaux
# CUDA par pas, d'une durée moyenne de 1,3 µs, très en dessous du coût de
# lancement d'un noyau. Ces lancements viennent d'ici : `predict_land` est
# appelé neuf fois par plan, chacun faisant 4 itérations de Newton x 3
# sous-pas x 4 étages RK, et chaque étage réévalue un modèle aérodynamique à
# interpolation par table — environ 5 700 opérations élémentaires par appel.
#
# `torch.compile` fusionne tout ça : -91 % d'opérations, à résultats
# numériquement identiques (écart maximal mesuré 5e-7, soit le bruit du
# float32). `dynamic=True` produit UN graphe valable pour toutes les tailles
# de lot : indispensable ici, le nombre d'environnements resetés changeant à
# chaque pas, une compilation par taille passerait son temps à recompiler.
_propagate_dispatch: dict = {}


def _rk4(position, velocity, spin, dt, params, *, coefficients=None):
    """Point d'entrée du pas RK4 : compilé si disponible, sinon direct."""

    return rk4_step(position, velocity, spin, dt, params, coefficients=coefficients)


def enable_compiled_flight(enabled: bool = True) -> bool:
    """Active (ou coupe) la version compilée. Renvoie l'état obtenu.

    Échouer ici ne doit jamais faire échouer un run : une compilation qui ne
    passe pas coûte de la vitesse, pas des résultats. On retombe alors
    silencieusement sur la version directe, en le signalant une fois.
    """

    if not enabled:
        _propagate_dispatch.clear()
        return False
    if _propagate_dispatch:
        return True
    try:
        # On compile les PROPAGATIONS entières, pas seulement le pas RK4 : le
        # pas s'y retrouve inliné avec la boucle de Newton et les sous-pas,
        # donc dans un seul graphe. Mesuré sur un appel complet au
        # planificateur : 140 397 opérations en direct, 15 405 en compilant le
        # seul pas RK4, 8 201 en compilant les propagations.
        _propagate_dispatch["height"] = torch.compile(_propagate_to_height, dynamic=True)
        _propagate_dispatch["x"] = torch.compile(_propagate_to_x, dynamic=True)
    except Exception as exc:  # pragma: no cover - dépend de la plateforme
        print(f"[physique] compilation indisponible ({exc}) : exécution directe.", flush=True)
        _propagate_dispatch.clear()
        return False
    return True


def integrate_flight_duration(
    position: torch.Tensor,
    velocity: torch.Tensor,
    spin: torch.Tensor,
    duration: torch.Tensor,
    *,
    substeps: int = 8,
    params: BallPhysicsParams = DEFAULT_BALL_PHYSICS,
    coefficients: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrate a different duration per batch element with a fixed graph.

    Event prediction uses this routine instead of hundreds of tiny Python-side
    time steps.  A handful of RK4 substeps is sufficient over the sub-second
    flights in this task; root refinement below removes the plane-crossing
    error.

    With no ``coefficients`` override, ``rk4_step`` re-evaluates the
    speed/spin-dependent drag and Magnus coefficients at every RK4 stage.  That
    evaluation dominates the cost here, but holding the coefficients fixed
    across a substep is a bad trade: it breaks RK4's error cancellation, so at
    matched wall-clock cost per-stage refresh measures ~30x more accurate than
    per-substep refresh with proportionally more substeps.  An explicit
    ``coefficients`` override freezes them for the whole call, for ablations.
    """

    if substeps < 1:
        raise ValueError("substeps must be positive")
    step_duration = duration / float(substeps)
    pos, vel, angvel = position, velocity, spin
    for _ in range(substeps):
        pos, vel, angvel = _rk4(
            pos,
            vel,
            angvel,
            step_duration,
            params,
            coefficients=coefficients,
        )
    return pos, vel, angvel


def _plane_value(value: float | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    plane = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    return torch.broadcast_to(plane, reference.shape[:-1])


def _propagate_to_height(
    position: torch.Tensor,
    velocity: torch.Tensor,
    spin: torch.Tensor,
    height: float | torch.Tensor,
    *,
    max_time: float = 1.5,
    descending: bool = True,
    params: BallPhysicsParams = DEFAULT_BALL_PHYSICS,
    root_iterations: int = 4,
    integration_substeps: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Propagate to a horizontal plane using batched shooting and Newton steps.

    Event accuracy is controlled explicitly by ``root_iterations`` and
    ``integration_substeps``.  There is no loop proportional to a simulation
    timestep or to ``max_time``.
    """

    if root_iterations < 1:
        raise ValueError("root_iterations must be positive")
    plane = _plane_value(height, position)
    gravity = params.gravity
    discriminant = velocity[..., 2] ** 2 + 2.0 * gravity * (position[..., 2] - plane)
    sqrt_discriminant = torch.sqrt(discriminant.clamp_min(0.0))
    if descending:
        event_time = (velocity[..., 2] + sqrt_discriminant) / gravity
    else:
        event_time = (velocity[..., 2] - sqrt_discriminant) / gravity

    initially_possible = discriminant >= 0.0
    event_time = event_time.nan_to_num(max_time).clamp(1.0e-5, max_time)
    out_pos, out_vel, out_spin = position, velocity, spin
    for iteration in range(root_iterations):
        out_pos, out_vel, out_spin = integrate_flight_duration(
            position,
            velocity,
            spin,
            event_time,
            substeps=integration_substeps,
            params=params,
        )
        if iteration + 1 < root_iterations:
            derivative = out_vel[..., 2]
            safe_derivative = torch.where(
                derivative.abs() > 1.0e-5, derivative, torch.full_like(derivative, -1.0e-5)
            )
            event_time = (
                event_time - (out_pos[..., 2] - plane) / safe_derivative
            ).nan_to_num(max_time).clamp(1.0e-5, max_time)

    direction_valid = out_vel[..., 2] < 0.0 if descending else out_vel[..., 2] > 0.0
    residual = (out_pos[..., 2] - plane).abs()
    hit = initially_possible & direction_valid & (event_time < max_time - 1.0e-5) & (residual < 3.0e-3)
    out_pos = out_pos.clone()
    out_pos[..., 2] = torch.where(hit, plane, out_pos[..., 2])
    return event_time, out_pos, out_vel, out_spin, hit


def _propagate_to_x(
    position: torch.Tensor,
    velocity: torch.Tensor,
    spin: torch.Tensor,
    target_x: float | torch.Tensor,
    *,
    max_time: float = 1.5,
    params: BallPhysicsParams = DEFAULT_BALL_PHYSICS,
    root_iterations: int = 4,
    integration_substeps: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Propagate to an x-plane with a fixed-cost batched root solve."""

    if root_iterations < 1:
        raise ValueError("root_iterations must be positive")
    plane = _plane_value(target_x, position)
    initial_delta = plane - position[..., 0]
    initial_direction = torch.sign(initial_delta)
    regular_velocity = velocity[..., 0].abs() > 1.0e-6
    event_time = torch.where(
        regular_velocity,
        initial_delta / torch.where(
            regular_velocity, velocity[..., 0], torch.ones_like(velocity[..., 0])
        ),
        torch.full_like(initial_delta, max_time),
    )
    event_time = event_time.nan_to_num(max_time).clamp(1.0e-5, max_time)
    out_pos, out_vel, out_spin = position, velocity, spin
    for iteration in range(root_iterations):
        out_pos, out_vel, out_spin = integrate_flight_duration(
            position,
            velocity,
            spin,
            event_time,
            substeps=integration_substeps,
            params=params,
        )
        if iteration + 1 < root_iterations:
            derivative = out_vel[..., 0]
            safe_derivative = torch.where(
                derivative.abs() > 1.0e-5,
                derivative,
                torch.where(initial_direction >= 0.0, 1.0e-5, -1.0e-5),
            )
            event_time = (
                event_time - (out_pos[..., 0] - plane) / safe_derivative
            ).nan_to_num(max_time).clamp(1.0e-5, max_time)

    direction_valid = initial_direction * out_vel[..., 0] > 0.0
    residual = (out_pos[..., 0] - plane).abs()
    hit = regular_velocity & direction_valid & (event_time < max_time - 1.0e-5) & (residual < 3.0e-3)
    out_pos = out_pos.clone()
    out_pos[..., 0] = torch.where(hit, plane, out_pos[..., 0])
    return event_time, out_pos, out_vel, out_spin, hit



def propagate_to_height(*args, **kwargs):
    """Façade : version compilée si disponible, sinon implémentation directe."""

    impl = _propagate_dispatch.get("height") or _propagate_to_height
    return impl(*args, **kwargs)


def propagate_to_x(*args, **kwargs):
    """Façade : version compilée si disponible, sinon implémentation directe."""

    impl = _propagate_dispatch.get("x") or _propagate_to_x
    return impl(*args, **kwargs)


def propagate_to_height_reference(
    position: torch.Tensor,
    velocity: torch.Tensor,
    spin: torch.Tensor,
    height: float | torch.Tensor,
    *,
    dt: float = 0.005,
    max_time: float = 2.0,
    descending: bool = True,
    params: BallPhysicsParams = DEFAULT_BALL_PHYSICS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Propagate to a horizontal plane and interpolate the impact state.

    Returns ``time, position, velocity, spin, hit_mask``.  States that do not
    reach the plane contain their state at ``max_time`` and ``hit_mask=False``.
    """

    plane = _plane_value(height, position)
    pos = position.clone()
    vel = velocity.clone()
    angvel = spin.clone()
    out_pos, out_vel, out_spin = pos.clone(), vel.clone(), angvel.clone()
    out_time = torch.full(pos.shape[:-1], max_time, dtype=pos.dtype, device=pos.device)
    active = torch.ones(pos.shape[:-1], dtype=torch.bool, device=pos.device)
    hit = torch.zeros_like(active)
    steps = max(1, math.ceil(max_time / dt))

    for step in range(steps):
        next_pos, next_vel, next_spin = rk4_step(pos, vel, angvel, dt, params)
        if descending:
            crossing = active & (pos[..., 2] >= plane) & (next_pos[..., 2] < plane)
        else:
            crossing = active & (pos[..., 2] <= plane) & (next_pos[..., 2] > plane)

        denominator = pos[..., 2] - next_pos[..., 2]
        valid_denominator = denominator.abs() > 1.0e-9
        safe_denominator = torch.where(valid_denominator, denominator, torch.ones_like(denominator))
        fraction = ((pos[..., 2] - plane) / safe_denominator).clamp(0.0, 1.0)
        fraction = torch.where(valid_denominator, fraction, torch.zeros_like(fraction))
        alpha = fraction.unsqueeze(-1)
        impact_pos = pos + alpha * (next_pos - pos)
        impact_pos[..., 2] = plane
        impact_vel = vel + alpha * (next_vel - vel)
        impact_spin = angvel + alpha * (next_spin - angvel)

        out_pos = torch.where(crossing.unsqueeze(-1), impact_pos, out_pos)
        out_vel = torch.where(crossing.unsqueeze(-1), impact_vel, out_vel)
        out_spin = torch.where(crossing.unsqueeze(-1), impact_spin, out_spin)
        impact_time = (step + fraction) * dt
        out_time = torch.where(crossing, impact_time, out_time)
        hit |= crossing
        active &= ~crossing

        pos = torch.where(active.unsqueeze(-1), next_pos, pos)
        vel = torch.where(active.unsqueeze(-1), next_vel, vel)
        angvel = torch.where(active.unsqueeze(-1), next_spin, angvel)
        out_pos = torch.where(active.unsqueeze(-1), pos, out_pos)
        out_vel = torch.where(active.unsqueeze(-1), vel, out_vel)
        out_spin = torch.where(active.unsqueeze(-1), angvel, out_spin)
        if not active.any():
            break

    return out_time, out_pos, out_vel, out_spin, hit


def propagate_to_x_reference(
    position: torch.Tensor,
    velocity: torch.Tensor,
    spin: torch.Tensor,
    target_x: float | torch.Tensor,
    *,
    dt: float = 0.005,
    max_time: float = 2.0,
    params: BallPhysicsParams = DEFAULT_BALL_PHYSICS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Propagate to a vertical x-plane and interpolate the crossing state."""

    plane = _plane_value(target_x, position)
    pos = position.clone()
    vel = velocity.clone()
    angvel = spin.clone()
    out_pos, out_vel, out_spin = pos.clone(), vel.clone(), angvel.clone()
    out_time = torch.full(pos.shape[:-1], max_time, dtype=pos.dtype, device=pos.device)
    active = torch.ones(pos.shape[:-1], dtype=torch.bool, device=pos.device)
    hit = torch.zeros_like(active)
    negative_direction = position[..., 0] > plane
    steps = max(1, math.ceil(max_time / dt))

    for step in range(steps):
        next_pos, next_vel, next_spin = rk4_step(pos, vel, angvel, dt, params)
        crosses_negative = (pos[..., 0] >= plane) & (next_pos[..., 0] < plane)
        crosses_positive = (pos[..., 0] <= plane) & (next_pos[..., 0] > plane)
        crossing = active & torch.where(negative_direction, crosses_negative, crosses_positive)

        denominator = next_pos[..., 0] - pos[..., 0]
        valid_denominator = denominator.abs() > 1.0e-9
        safe_denominator = torch.where(valid_denominator, denominator, torch.ones_like(denominator))
        fraction = ((plane - pos[..., 0]) / safe_denominator).clamp(0.0, 1.0)
        fraction = torch.where(valid_denominator, fraction, torch.zeros_like(fraction))
        alpha = fraction.unsqueeze(-1)
        crossing_pos = pos + alpha * (next_pos - pos)
        crossing_pos[..., 0] = plane
        crossing_vel = vel + alpha * (next_vel - vel)
        crossing_spin = angvel + alpha * (next_spin - angvel)

        out_pos = torch.where(crossing.unsqueeze(-1), crossing_pos, out_pos)
        out_vel = torch.where(crossing.unsqueeze(-1), crossing_vel, out_vel)
        out_spin = torch.where(crossing.unsqueeze(-1), crossing_spin, out_spin)
        crossing_time = (step + fraction) * dt
        out_time = torch.where(crossing, crossing_time, out_time)
        hit |= crossing
        active &= ~crossing

        pos = torch.where(active.unsqueeze(-1), next_pos, pos)
        vel = torch.where(active.unsqueeze(-1), next_vel, vel)
        angvel = torch.where(active.unsqueeze(-1), next_spin, angvel)
        out_pos = torch.where(active.unsqueeze(-1), pos, out_pos)
        out_vel = torch.where(active.unsqueeze(-1), vel, out_vel)
        out_spin = torch.where(active.unsqueeze(-1), angvel, out_spin)
        if not active.any():
            break

    return out_time, out_pos, out_vel, out_spin, hit


def table_impact(
    velocity: torch.Tensor,
    spin: torch.Tensor,
    normal: torch.Tensor | None = None,
    params: BallPhysicsParams = DEFAULT_BALL_PHYSICS,
    friction: float | torch.Tensor | None = None,
    restitution_scale: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Apply an instantaneous frictional impact with a stationary table."""

    if normal is None:
        normal = torch.zeros_like(velocity)
        normal[..., 2] = 1.0
    normal = _normalize(torch.broadcast_to(normal, velocity.shape))
    normal_velocity = torch.sum(velocity * normal, dim=-1)
    incoming = normal_velocity < 0.0

    restitution = (
        params.table_restitution_base
        + params.table_restitution_velocity_slope * normal_velocity
    )
    restitution = restitution * torch.as_tensor(
        restitution_scale, dtype=velocity.dtype, device=velocity.device
    )
    restitution = restitution.clamp(params.table_restitution_min, params.table_restitution_max)

    tangential_contact_velocity = (
        velocity
        - normal_velocity.unsqueeze(-1) * normal
        + params.radius * torch.cross(normal, spin, dim=-1)
    )
    tangential_speed = torch.linalg.vector_norm(tangential_contact_velocity, dim=-1)
    stick_fraction = 1.0 / (1.0 + params.inertia_ratio)
    if friction is None:
        friction = params.table_friction
    friction_tensor = torch.as_tensor(friction, dtype=velocity.dtype, device=velocity.device)
    slide_fraction = (
        friction_tensor
        * (1.0 + restitution)
        * normal_velocity.abs()
        / tangential_speed.clamp_min(1.0e-8)
    )
    alpha = torch.minimum(torch.full_like(slide_fraction, stick_fraction), slide_fraction)

    delta_velocity = (
        -(1.0 + restitution).unsqueeze(-1) * normal_velocity.unsqueeze(-1) * normal
        - alpha.unsqueeze(-1) * tangential_contact_velocity
    )
    delta_spin = (
        params.inertia_ratio
        * alpha.unsqueeze(-1)
        / params.radius
        * torch.cross(normal, tangential_contact_velocity, dim=-1)
    )
    out_velocity = torch.where(incoming.unsqueeze(-1), velocity + delta_velocity, velocity)
    out_spin = torch.where(incoming.unsqueeze(-1), spin + delta_spin, spin)
    return out_velocity, out_spin, {
        "incoming": incoming,
        "restitution": restitution,
        "tangential_contact_speed": tangential_speed,
        "sliding": slide_fraction < stick_fraction,
    }


def racket_impact(
    ball_velocity: torch.Tensor,
    ball_spin: torch.Tensor,
    racket_velocity: torch.Tensor,
    racket_spin: torch.Tensor,
    normal: torch.Tensor,
    contact_offset: torch.Tensor | None = None,
    params: BallPhysicsParams = DEFAULT_BALL_PHYSICS,
    grip_scale: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the identified racket/ball impact model in world coordinates.

    ``contact_offset`` points from the racket body origin to the contact point.
    The normal is automatically flipped to face the incoming ball, which makes
    the function insensitive to which physical face of the paddle was used.
    """

    if contact_offset is None:
        contact_offset = torch.zeros_like(ball_velocity)
    normal = _normalize(torch.broadcast_to(normal, ball_velocity.shape))
    racket_contact_velocity = racket_velocity + torch.cross(racket_spin, contact_offset, dim=-1)

    relative_center_velocity = ball_velocity - racket_contact_velocity
    normal_sign = torch.where(
        torch.sum(relative_center_velocity * normal, dim=-1, keepdim=True) > 0.0,
        -torch.ones_like(normal[..., :1]),
        torch.ones_like(normal[..., :1]),
    )
    normal = normal * normal_sign

    ball_contact_velocity = ball_velocity + params.radius * torch.cross(normal, ball_spin, dim=-1)
    relative_velocity = ball_contact_velocity - racket_contact_velocity
    normal_velocity = torch.sum(relative_velocity * normal, dim=-1)
    incoming = normal_velocity < 0.0
    tangential_velocity = relative_velocity - normal_velocity.unsqueeze(-1) * normal
    tangential_speed = torch.linalg.vector_norm(tangential_velocity, dim=-1)

    normal_restitution = (
        params.racket_normal_restitution_base
        + params.racket_normal_restitution_velocity_slope * normal_velocity.abs()
    ).clamp(params.racket_normal_restitution_min, params.racket_normal_restitution_max)
    tangential_restitution = (
        params.racket_tangential_restitution_base
        + params.racket_tangential_restitution_velocity_slope * tangential_speed
    ).clamp(params.racket_tangential_restitution_min, params.racket_tangential_restitution_max)
    grip_scale_tensor = torch.as_tensor(grip_scale, dtype=ball_velocity.dtype, device=ball_velocity.device)
    unconstrained_alpha = (1.0 + tangential_restitution) / (1.0 + params.inertia_ratio)
    friction = params.racket_friction * grip_scale_tensor
    coulomb_alpha = (
        friction
        * (1.0 + normal_restitution)
        * normal_velocity.abs()
        / tangential_speed.clamp_min(1.0e-8)
    )
    alpha = torch.minimum(unconstrained_alpha, coulomb_alpha)

    delta_velocity = (
        -(1.0 + normal_restitution).unsqueeze(-1) * normal_velocity.unsqueeze(-1) * normal
        - alpha.unsqueeze(-1) * tangential_velocity
    )
    delta_spin = (
        params.inertia_ratio
        * alpha.unsqueeze(-1)
        / params.radius
        * torch.cross(normal, tangential_velocity, dim=-1)
    )
    out_velocity = ball_velocity + delta_velocity
    out_spin = ball_spin + delta_spin
    normal_spin = torch.sum(out_spin * normal, dim=-1, keepdim=True) * normal
    out_spin = out_spin + (params.racket_normal_spin_retention - 1.0) * normal_spin

    out_velocity = torch.where(incoming.unsqueeze(-1), out_velocity, ball_velocity)
    out_spin = torch.where(incoming.unsqueeze(-1), out_spin, ball_spin)
    return out_velocity, out_spin, {
        "incoming": incoming,
        "normal": normal,
        "normal_restitution": normal_restitution,
        "tangential_restitution": tangential_restitution,
        "relative_contact_velocity": relative_velocity,
        "tangential_impulse_fraction": alpha,
        "coulomb_limit": coulomb_alpha,
        "friction_limited": coulomb_alpha < unconstrained_alpha,
    }
