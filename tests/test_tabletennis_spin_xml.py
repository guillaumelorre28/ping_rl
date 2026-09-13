"""Structural checks for spin-relevant MuJoCo model properties."""

from pathlib import Path

import mujoco
import pytest
from planner import (
    DEFAULT_BALL_CONTACT_HEIGHT,
    MIN_RETURN_NET_HEIGHT,
    MIN_SERVE_NET_HEIGHT,
    NET_TOP_HEIGHT,
)
from tabletennis_env import _set_multiccd, apply_ball_inertia


@pytest.fixture(scope="module")
def tabletennis_model():
    xml_path = Path(__file__).resolve().parents[1] / "tabletennis.xml"
    return mujoco.MjModel.from_xml_path(str(xml_path))


def test_ball_uses_thin_shell_inertia(tabletennis_model):
    xml_path = Path(__file__).resolve().parents[1] / "tabletennis.xml"
    spec = mujoco.MjSpec.from_file(str(xml_path))
    assert spec.body("pingpong").mass == pytest.approx(2.7e-3)
    assert spec.body("pingpong").inertia.tolist() == pytest.approx([7.2e-7] * 3)

    original_inertia = tabletennis_model.body_inertia.copy()
    ball_id = tabletennis_model.body("pingpong").id
    ball_dof = tabletennis_model.body_dofadr[ball_id]
    original_rotational_invweight = tabletennis_model.dof_invweight0[ball_dof + 3 : ball_dof + 6].copy()
    apply_ball_inertia(tabletennis_model)
    assert tabletennis_model.body_inertia[ball_id].tolist() == pytest.approx([7.2e-7] * 3)
    assert (
        tabletennis_model.dof_invweight0[ball_dof + 3 : ball_dof + 6]
        > original_rotational_invweight
    ).all()
    unchanged_body_ids = [index for index in range(tabletennis_model.nbody) if index != ball_id]
    assert (
        tabletennis_model.body_inertia[unchanged_body_ids]
        == original_inertia[unchanged_body_ids]
    ).all()


def test_ball_contacts_have_dedicated_pairs(tabletennis_model):
    ball_id = tabletennis_model.geom("pingpong").id
    required_other_geoms = {
        tabletennis_model.geom("coll_own_half").id,
        tabletennis_model.geom("coll_opponent_half").id,
        tabletennis_model.geom("pad").id,
    }
    paired_with_ball = set()
    for geom1, geom2 in zip(tabletennis_model.pair_geom1, tabletennis_model.pair_geom2, strict=True):
        if geom1 == ball_id:
            paired_with_ball.add(int(geom2))
        elif geom2 == ball_id:
            paired_with_ball.add(int(geom1))

    assert required_other_geoms <= paired_with_ball


def test_angular_velocity_sensors_are_available(tabletennis_model):
    assert tabletennis_model.sensor("pingpong_angvel_sensor").dim[0] == 3
    assert tabletennis_model.sensor("paddle_angvel_sensor").dim[0] == 3


def test_planner_contact_height_matches_table_and_ball_geometry(tabletennis_model):
    table_id = tabletennis_model.geom("coll_own_half").id
    ball_id = tabletennis_model.geom("pingpong").id
    table_surface = tabletennis_model.geom_pos[table_id, 2] + tabletennis_model.geom_size[table_id, 2]
    ball_radius = tabletennis_model.geom_size[ball_id, 0]

    assert DEFAULT_BALL_CONTACT_HEIGHT == pytest.approx(table_surface + ball_radius)


def test_net_thresholds_stay_above_the_modelled_tape(tabletennis_model):
    """Keep the crossing gates tied to the net geometry they are derived from.

    The gates are plain numbers in the planner, so a change to `coll_net` would
    otherwise silently move them below the tape and start scoring net balls as
    legal returns.
    """

    net_id = tabletennis_model.geom("coll_net").id
    tape_height = (
        tabletennis_model.geom_pos[net_id, 2] + tabletennis_model.geom_size[net_id, 2]
    )

    assert NET_TOP_HEIGHT == pytest.approx(tape_height)
    assert MIN_SERVE_NET_HEIGHT > tape_height
    assert MIN_RETURN_NET_HEIGHT >= MIN_SERVE_NET_HEIGHT


def test_multiccd_is_cleared_whatever_mujoco_calls_it():
    """Le flag doit partir sous les deux orthographes, selon la version.

    MuJoCo a déplacé multi-contact CCD d'un *enable* (`mjENBL_MULTICCD`,
    jusqu'en 3.4) vers un *disable* (`mjDSBL_MULTICCD`, dès 3.13). La révision
    de MuJoCo-Warp épinglée ignore les deux et refuse `put_model` sur le bit
    inconnu. Écrire une seule orthographe marchait dans le venv de
    développement et faisait planter le conteneur, ce qui n'a été découvert
    qu'en louant un GPU.
    """

    xml_path = Path(__file__).resolve().parents[1] / "tabletennis.xml"
    spec = mujoco.MjSpec.from_file(str(xml_path))
    _set_multiccd(spec, False)
    model = spec.compile()

    enable_bit = getattr(mujoco.mjtEnableBit, "mjENBL_MULTICCD", None)
    disable_bit = getattr(mujoco.mjtDisableBit, "mjDSBL_MULTICCD", None)
    assert enable_bit is not None or disable_bit is not None, (
        "aucune des deux orthographes n'existe : le test ne vérifie plus rien"
    )
    if enable_bit is not None:
        assert not model.opt.enableflags & int(enable_bit)
    if disable_bit is not None:
        assert not model.opt.disableflags & int(disable_bit)
