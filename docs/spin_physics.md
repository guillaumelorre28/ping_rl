# Spin-aware ball physics

The simulation and the paddle planner share the same PyTorch model in
`ball_physics.py`. This is intentional: a planner using gravity-only ballistics
would generate systematically wrong racket commands once spin is enabled.

## State and conventions

The ball state is `(position, velocity, angular_velocity)` in world coordinates.
Policy commands use trajectory coordinates:

- component 0: topspin (positive bends the flight down),
- component 1: sidespin,
- component 2: corkscrew spin around the direction of travel.

`spin.reward_axis_weights` selects which of these three components contribute
to the spin objective. For example, `[1, 0, 0]` learns only topspin/backspin and
`[0, 1, 0]` learns only sidespin.

The default uniform training range controls topspin/backspin and sidespin. Its
corkscrew range is zero because an isotropic single-point contact cannot impose
an arbitrary torque around the contact normal. Incoming corkscrew is still
randomized as a disturbance, and fixed non-zero commands remain available for
experiments with a richer identified contact model.

## Flight

At every 2 ms MuJoCo substep, the environment applies aerodynamic drag and
Magnus force:

```text
F_drag   = -0.5 rho A C_d |v| v
F_magnus = rho V C_m (omega x v)
```

The default `conti2026` model uses the fitted piecewise tables from Conti et al.:
`C_d` depends on speed and spin ratio `r|omega|/|v|`, while `C_m` depends on
speed and spin magnitude. This supersedes the earlier Nature baseline
`C_m = 0.1 r|omega|/|v| - 0.001`: substituting that baseline in the force law
makes its main Magnus term scale quadratically with spin. A `constant` model is
retained for controlled ablations; its force scales as `|omega||v|`, consistent
with the compact TT4D formulation.

Gravity remains native to MuJoCo. The planner uses the same equations. Its
plane-event solver estimates the crossing time ballistically, refines it with
Newton iterations, and integrates each candidate with a fixed small number of
RK4 substeps. Runtime therefore does not grow as `max_time / dt`. Aerodynamic
coefficients are evaluated from the current state at every RK4 stage, matching
the state-dependent force used by the environment. The defaults use four root
iterations and three integration substeps; both are exposed under
`spin.planner` as `event_root_iterations` and `event_integration_substeps`.
Three root iterations reject about 15% of heavy-backspin flights, so four is
the minimum safe value; a fifth changes nothing. Holding the coefficients fixed
across a substep instead of refreshing them per RK4 stage is a bad trade -- it
breaks RK4's error cancellation, and at matched wall-clock cost measures around
30x less accurate even with proportionally more substeps.

The ball inertia in `tabletennis.xml` is the thin-shell value
`2/3 m r^2 = 7.2e-7 kg m^2`, rather than the former `1e-4` placeholder. Because
MuJoCo's `boundinertia` compiler option is global, the environment reapplies
this declared value after compilation and calls `mj_setConst`; derived fields
such as `dof_invweight0` are therefore recomputed before conversion to Warp.

### Scope of the published models

The implementation reproduces the published aerodynamic coefficient tables
and the analytic racket-impact backbone (including the reported restitution
and spin-retention fits). It deliberately does not claim to reproduce the
paper's learned racket residual network: no transferable trained weights are
provided, and that residual is tied to a particular robot, rubber, sensing
pipeline, and impact dataset. The table-impact Lasso residual is likewise not
enabled by default; the Coulomb map is the conservative, equipment-independent
baseline. Both residuals should be fitted from this project's measurements and
validated on held-out impacts before being introduced as optional corrections.

## Contacts

On the first contact substep, the environment replaces the native post-contact
ball velocity and angular velocity with the analytic impact map:

- table: velocity-dependent normal restitution plus a Coulomb-limited
  tangential impulse coupled to angular velocity;
- racket: moving-contact velocity (linear plus angular racket motion),
  velocity-dependent normal/tangential restitution, translation-spin coupling,
  and a Coulomb bound on tangential impulse. In particular,
  `alpha = min((1 + e_t)/(1 + kappa), mu_r(1 + e_n)|v_n|/|v_t|)`, so a grazing
  impact cannot create finite spin without a supporting normal impulse.

The native contact is not a spin-transfer substitute for the analytic map, and
the gap is large. On the project model at 2 ms with the ball arriving at 5 m/s,
the analytic racket map produces 107.9 rad/s of spin where the native contact
produces 1.15 rad/s -- about 94x less. On the table bounce with 150 rad/s of
incoming topspin, the analytic map adds 60 rad/s of spin and brakes the
horizontal velocity by 0.80 m/s, against +0.73 rad/s and 0.01 m/s natively.
`spin.analytic_contact_override: false` is therefore a near-zero-spin-transfer
condition rather than a like-for-like contact comparison; read that ablation
accordingly.

Those figures come from MuJoCo on CPU, but the training loop steps MuJoCo-Warp.
The two engines are pinned against each other on the same scene at 2, 1 and
0.5 ms: they agree to within 2e-3 m/s and 5e-2 rad/s, so the CPU
characterization transfers. That check is marked `slow` because it compiles
Warp kernels on first run.

The native solver still handles contact geometry and the first arm/racket
response. When the ball state is replaced, the difference from the native
linear and angular momenta is applied with opposite sign to the racket free
body; subsequent grasp contacts can transmit that correction to the hand.
Contact edges use a two-substep release hysteresis, so one-frame sensor chatter
cannot apply the map twice. Set
`spin.analytic_contact_override: false` for a native-contact ablation.

The target landing height is 0.815 m: 0.795 m table surface height plus the
0.020 m ball radius.

## Planner and reward

`planner.plan_spin_return` first predicts the incoming table bounce and state at
the robot hit plane. It solves a nonlinear shooting problem for the outgoing
velocity with a finite-difference 2x2 landing Jacobian, verifies net clearance,
converts the requested spin to world coordinates, inverts the friction-limited
racket impact, then corrects the landing point using the spin actually
achievable at contact. Its pose target includes the 40 mm center-to-center
contact offset (20 mm ball radius plus 20 mm racket half-thickness).

Net clearance uses two named gates in `planner.py` rather than repeated
literals: `MIN_RETURN_NET_HEIGHT = 1.0` for the robot's return (shared by the
planner's validity check and the reward, so a plan the planner accepts is one
the reward can score) and `MIN_SERVE_NET_HEIGHT = 0.98` for accepting a
resampled incoming serve. Both sit above `NET_TOP_HEIGHT = 0.9475`, the top of
the `coll_net` box, and a model test pins that relation so a change to the net
geometry cannot silently move a gate below the tape.

The two correction counters are solved values rather than tuning surface, so
they stay function defaults instead of being exposed in the YAML. Measured
against 0.5 ms integration, `shot_correction_iterations = 1` finds no
net-clearing trajectory for 27% of the batch and
`impact_correction_iterations = 1` leaves a 248 mm landing error at p95; a
second impact iteration still leaves 18 mm at p95 and 221 mm worst case, while
the shipped `2` and `3` give 1.0 mm at p95 and 20 mm worst case. Raising either
further changes nothing. `tests/test_spin_planner.py` pins this by
re-integrating the planned shot finely, so lowering either counter fails the
suite.

The required contact-point velocity is split between body translation and
`omega_racket x contact_offset`. The split is exact at the contact point and
bounded by configurable linear/angular limits, but it is not unique: any twist
satisfying `v_body + omega x r = v_contact` produces the same impact, so
`wrist_fraction` only picks one point on that continuum.

The command reward therefore scores `paddle_contact_vel_err` on the contact
point itself rather than on either half of the split. Scoring `omega` would
constrain a null mode; scoring `v_body` alone -- which the earlier
`paddle_vel_err` did -- silently ignores the rotational part, and that part is
not small: measured over 2048 plans, `omega x r` carries a median 25% of the
contact velocity and 37% at p95, so a policy holding the wrist still while
matching `v_body` used to collect the full term for a materially different
shot. The plan exposes `contact_velocity` and `contact_offset_local` (the lever
arm in the paddle body frame, `[-0.07, 0, -0.02]` for this model) so the
environment can rebuild the contact velocity once the paddle has moved.

The same reasoning already governs orientation: the paddle loss aligns only the
face normal, leaving roll and the equivalent opposite face unconstrained.

### Reachable spin commands

A single tangential impulse bounds the spin change by `(kappa / r) |delta v|`,
so the commanded box is not uniformly attainable and its corners are not
attainable at all. Measured over 2048 plans, the worst-axis error between the
commanded and the achieved trajectory-frame spin grows with the commanded
magnitude:

| `|command|` (rad/s) | median | p95 |
|---|---|---|
| 0-150 | 0.7 | 2.8 |
| 150-250 | 1.7 | 5.5 |
| 250-325 | 2.1 | 8.3 |
| 325-500 | 2.8 | 45.6 |

The shortfall is in magnitude, not direction: the spin axis stays within 13
degrees at p95, so a requested effect is never inverted. The default
`spin.target_range` box spans `+/-400` topspin by `+/-250` sidespin, whose
corner is `sqrt(400^2 + 250^2) = 471` rad/s. Sampling it uniformly puts 27.7%
of commands above 325 rad/s and 6.9% above 400 rad/s, where the reference plan
itself misses the command by roughly the reward's `sigma` of 80 rad/s. Backspin
is the harder direction because its smaller `|delta v|` leaves a smaller spin
budget.

No fixed norm cap makes every command reachable: the budget is
`(kappa / r) |delta v|`, and `delta v` depends on the incoming velocity and
spin and on the requested landing point, so the feasible set differs per
sample. Narrowing the box only reduces how often a command is infeasible. The
options are therefore to project each command onto its own feasible set --
which needs the planner's `delta v`, so it has to happen after the shooting
solve -- or to accept a stated tolerance and note that the reward is softened
rather than broken in the infeasible tail. The current configuration takes the
second option implicitly; making it explicit is preferable.

The actor observes ball spin and the trajectory-frame spin command. The critic
also observes the active drag, Magnus, table-friction, and racket-grip scales;
they are randomized only when `enable_domain_randomization: true`.
The spin reward is evaluated from the first post-racket state and is gated by a
legal opponent-side return. A separate landing reward prevents a policy from
trading placement away for spin magnitude.

When enabled, `spin.curriculum` linearly grows incoming and requested spin from
`initial_scale` to full amplitude by `full_fraction` of the PPO iterations.
Evaluation bypasses this scaling. `spin.shaping` independently decays the dense
planner-imitation terms while leaving legal-return, placement, and requested
spin rewards unchanged.

## Ablations

The YAML exposes independent switches for drag, Magnus, incoming spin, the
spin-aware planner, the spin reward, and analytic contact override. Setting
`spin.enabled: false` is the strict legacy condition: it restores gravity-only
planning, native contacts, the original target offset/orientation calculation,
and the original actor/critic and reward interfaces. `enable_multiccd` defaults
to false because the repository's pinned MuJoCo-Warp revision does not support
it.

## Calibration protocol

The coefficients in `BallPhysicsParams` are published priors, not a calibration
of this repository's equipment. They should be identified against measured or
high-fidelity trajectories before sim-to-real use:

1. fit drag/Magnus parameters on free-flight trajectories with known initial
   velocity and spin;
2. fit table restitution/friction on pre/post-bounce linear and angular states;
3. fit racket normal/tangential restitution and spin retention on held-out
   impacts covering racket speed, orientation, and brushing direction;
4. report position, velocity, and spin errors on held-out rollouts rather than
   only training loss.

The automated regressions check force signs, coefficient ranges, RK4/event
convergence, sliding versus sticking, grazing-contact friction limits,
off-center racket twist, net/kinematic validity, and a 128-shot planner batch.
The default `conti2026` event solver is compared with a 0.5 ms reference over
the complete configured topspin/backspin range. `tests/test_spin_mujoco_reference.py`
uses the actual project XML and contact parameters at 2, 1, and 0.5 ms. It
records the quantitative gap between native and analytic contact rather than
claiming agreement between the two. These checks validate implementation
consistency; they do not replace fitting on measurements from the actual ball,
rubber, and table.

## References

- Dürr et al., *Outplaying elite table tennis players with an autonomous
  robot*, Nature (2026), article `s41586-026-10338-5`.
- Cai et al., *TT4D: A Pipeline and Dataset for Table Tennis 4D Reconstruction
  From Monocular Videos*, arXiv:2605.01234v2,
  <https://arxiv.org/abs/2605.01234v2>.
- Conti et al., *Physics Models for Sim-to-Real Transfer in Professional-Level
  Robot Table Tennis*, arXiv:2606.28805v2,
  <https://arxiv.org/abs/2606.28805v2>.
