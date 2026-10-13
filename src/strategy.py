import math
from pathlib import Path

import pandas as pd

# Fitted on 2022-2024 laps only (05_model.ipynb, section 5E)
WEAR_MEDIUM = 0.0361  # seconds lost per lap of MEDIUM tyre age
WEAR_HARD = 0.0314    # seconds lost per lap of HARD tyre age
PACE_GAP = 0.0451     # fresh MEDIUM is faster than fresh HARD by this many s/lap
MIN_STINT = 8         # shortest allowed stint, laps

PIT_LOSS_PATH = Path(__file__).parent / "data" / "pit_loss.csv"


def optimal_pit_lap(race_laps: int) -> float:
    if race_laps < 2 * MIN_STINT:
        raise ValueError(f"race_laps must be at least {2 * MIN_STINT}")

    total_wear = WEAR_MEDIUM + WEAR_HARD
    lap = (WEAR_HARD * race_laps + PACE_GAP) / total_wear

    earliest = MIN_STINT
    latest = race_laps - MIN_STINT
    if lap < earliest:
        lap = earliest
    if lap > latest:
        lap = latest

    return lap


def pit_window(race_laps: int, tolerance_sec: float = 1.0) -> tuple[int, int]:
    best_lap = optimal_pit_lap(race_laps)

    total_wear = WEAR_MEDIUM + WEAR_HARD
    half_width = math.sqrt(2 * tolerance_sec / total_wear)

    start = round(best_lap - half_width)
    end = round(best_lap + half_width)

    earliest = MIN_STINT
    latest = race_laps - MIN_STINT
    if start < earliest:
        start = earliest
    if end > latest:
        end = latest

    return start, end


def load_pit_loss(path: Path = PIT_LOSS_PATH) -> dict[str, float]:
    table = pd.read_csv(path)

    pit_loss = {}
    for _, row in table.iterrows():
        event = row["Event"]
        loss = float(row["PitLoss"])
        pit_loss[event] = loss

    return pit_loss


def plan_one_stop(event: str, race_laps: int, pit_loss: dict[str, float]) -> dict:
    if event not in pit_loss:
        raise KeyError(f"Unknown event: {event}")

    best_lap = optimal_pit_lap(race_laps)
    start, end = pit_window(race_laps)

    result = {
        "event": event,
        "race_laps": race_laps,
        "strategy": "MEDIUM -> HARD",
        "optimal_pit_lap": round(best_lap),
        "window_start": start,
        "window_end": end,
        "pit_loss_sec": pit_loss[event],
    }
    return result