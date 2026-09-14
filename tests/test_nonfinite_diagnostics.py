"""Une divergence de la physique doit se voir, pas se faire absorber.

Trois `nan_to_num` muets — deux sur les observations, un sur la récompense —
faisaient passer un environnement dont l'état avait divergé pour un
environnement ordinaire à récompense nulle. Sur un run réel du 13 septembre
2026, à l'itération 18, trois termes sont passés en NaN et n'en sont jamais
revenus : `rel_pos_err`, `fin_open` et `paddle_pos_err`, tous trois lisant des
positions, tandis que les termes angulaires restaient sains. Le total n'a
baissé que de 8 %, donc rien n'a alerté ; mais 4,2 points par pas étaient
gagnés sans être journalisés, et ClearML affichant un NaN comme un zéro, les
courbes semblaient simplement plates.

Ces tests verrouillent les deux moitiés de la correction : la moyenne qui ne
se laisse plus effacer par une entrée corrompue, et le compte qui date
l'incident.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from on_policy_runner import finite_mean  # noqa: E402


def test_a_single_nan_no_longer_erases_the_whole_series():
    """C'est le défaut observé : une entrée sur mille effaçait la courbe."""

    values = torch.tensor([1.0, 2.0, 3.0, float("nan")])
    value, dropped = finite_mean(values)
    assert value == pytest.approx(2.0)
    assert dropped == 1

    # L'infini vient du même mécanisme (une translation qui diverge) et doit
    # être écarté de la même façon.
    value, dropped = finite_mean(torch.tensor([4.0, float("inf"), -float("inf")]))
    assert value == pytest.approx(4.0)
    assert dropped == 2


def test_lists_and_tensors_behave_identically():
    """Le runner agrège des deques de flottants ET des tenseurs."""

    for build in (list, torch.tensor):
        value, dropped = finite_mean(build([10.0, float("nan"), 20.0]))
        assert float(value) == pytest.approx(15.0)
        assert dropped == 1


def test_all_corrupt_reports_nothing_rather_than_a_plausible_zero():
    """Sans donnée saine, mieux vaut ne rien tracer qu'un zéro crédible.

    Un point à zéro se lit comme une mesure ; une absence se lit comme une
    absence. C'est toute la différence entre les deux modes d'échec.
    """

    value, dropped = finite_mean(torch.tensor([float("nan"), float("nan")]))
    assert value is None and dropped == 2

    value, dropped = finite_mean([])
    assert value is None and dropped == 0


def test_healthy_input_is_untouched():
    value, dropped = finite_mean(torch.tensor([1.0, 2.0, 3.0]))
    assert float(value) == pytest.approx(2.0)
    assert dropped == 0
    assert math.isfinite(float(value))
