import pytest

from src.strategy import optimal_pit_lap
from src.strategy import optimal_pit_lap, pit_window
from src.strategy import optimal_pit_lap, pit_window, load_pit_loss, plan_one_stop

def test_optimal_pit_lap_monza():
    assert optimal_pit_lap(53) == pytest.approx(25.32, abs=0.01)


def test_optimal_pit_lap_too_short_race():
    with pytest.raises(ValueError):
        optimal_pit_lap(10)


def test_pit_window_monza():
    assert pit_window(53) == (20, 31)


def test_pit_window_contains_optimal_lap():
    start, end = pit_window(53)
    assert start <= optimal_pit_lap(53) <= end


def test_pit_window_short_race_is_clipped():
    assert pit_window(20) == (8, 12)

def test_load_pit_loss_real_file():
    pit_loss = load_pit_loss()
    assert len(pit_loss) == 24
    assert pit_loss["Monaco Grand Prix"] == pytest.approx(19.08, abs=0.01)


def test_load_pit_loss_values_are_floats():
    pit_loss = load_pit_loss()
    for loss in pit_loss.values():
        assert isinstance(loss, float)


def test_plan_one_stop_returns_full_plan():
    fake_pit_loss = {"Test Grand Prix": 20.0}
    plan = plan_one_stop("Test Grand Prix", 53, fake_pit_loss)
    assert plan == {
        "event": "Test Grand Prix",
        "race_laps": 53,
        "strategy": "MEDIUM -> HARD",
        "optimal_pit_lap": 25,
        "window_start": 20,
        "window_end": 31,
        "pit_loss_sec": 20.0,
    }


def test_plan_one_stop_unknown_event():
    with pytest.raises(KeyError):
        plan_one_stop("Monza", 53, {"Test Grand Prix": 20.0})