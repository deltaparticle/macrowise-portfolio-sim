import numpy as np

from engine.optimizer import UserRequest, _resolve_universe
from engine.models._common import project_to_simplex


class DummyTag:
    def __init__(self, sectors=None, themes=None, sizes=None, styles=None, exclude=False):
        self.sectors = sectors or []
        self.themes = themes or []
        self.sizes = sizes or []
        self.styles = styles or []
        self.exclude = exclude


def test_resolve_universe_falls_back_to_any_match_when_all_match_fails(monkeypatch):
    fake_map = {
        "it_index": DummyTag(sectors=["IT"]),
        "bank_index": DummyTag(sectors=["Banks"]),
        "other_index": DummyTag(sectors=["Auto"]),
    }
    monkeypatch.setattr("engine.optimizer.build_map", lambda: fake_map)

    req = UserRequest(sectors=["IT", "Banks"], sector_match_all=True)
    universe = _resolve_universe(req)

    assert universe == ["it_index", "bank_index"]


def test_project_to_simplex_respects_bounds_and_budget():
    raw = np.array([0.8, 0.4, -0.2, 1.4])
    proj = project_to_simplex(raw, w_min=0.0, w_max=0.3, budget=1.0)

    assert proj.shape == raw.shape
    assert np.isclose(proj.sum(), 1.0)
    assert np.all(proj >= 0.0)
    assert np.all(proj <= 0.3 + 1e-8)
