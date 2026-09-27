import fastf1
import pandas as pd
from pathlib import Path
from fastf1.exceptions import RateLimitExceededError

from src.clean import clean_laps
fastf1.set_log_level("WARNING")


def load_race(year: int, round_number: int) -> pd.DataFrame:
    """Load one race, add race identifiers, return clean laps."""
    session = fastf1.get_session(year, round_number, "R")
    session.load(telemetry=False, weather=False, messages=True)
    laps = clean_laps(session.laps)

    laps["Year"] = year
    laps["Round"] = round_number
    laps["Event"] = session.event["EventName"]
    return laps

def load_season(year: int, max_rounds: int | None = None) -> pd.DataFrame:
    """Load clean laps for every race of a season."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    rounds = schedule["RoundNumber"].tolist()
    if max_rounds is not None:
        rounds = rounds[:max_rounds]
    races = []
    for rnd in rounds:
        try:
            laps = load_race(year, rnd)
            races.append(laps)
            print(f"{year} R{rnd}: {len(laps)} laps")
        except Exception as e:
            print(f"{year} R{rnd}: FAILED - {e}")

    return pd.concat(races, ignore_index=True)

def ingest_season(year: int, out_dir: Path) -> bool:
    """Save each race of a season to its own file. Skip races already saved."""
    out_dir.mkdir(parents=True, exist_ok=True)
    schedule = fastf1.get_event_schedule(year, include_testing=False)

    for rnd in schedule["RoundNumber"]:
        path = out_dir / f"{year}_R{rnd:02d}.parquet"
        if path.exists():
            continue

        try:
            laps = load_race(year, rnd)
            laps.to_parquet(path)
            print(f"{year} R{rnd}: {len(laps)} laps saved")
        except RateLimitExceededError:
            print(f"Rate limit hit at {year} R{rnd}. Rerun in ~1 hour.")
            return False
        except Exception as e:
            print(f"{year} R{rnd}: FAILED - {e}")

    return True