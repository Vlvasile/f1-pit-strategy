import pandas as pd

def clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (laps["LapNumber"] > 1)              # not the first lap (standing start)
        & laps["PitInTime"].isna()           # not a pit-in lap
        & laps["PitOutTime"].isna()          # not a pit-out lap
        & (laps["TrackStatus"] == "1")       # green flag only (no SC/VSC/yellow)
        & laps["IsAccurate"]                 # FastF1 trusts the timing
        & ~laps["Deleted"]                   # not deleted (track limits)
        & ~laps["FastF1Generated"]           # real lap, not reconstructed
    )
    clean = laps[mask].copy()
    clean["LapTimeSec"] = clean["LapTime"].dt.total_seconds()
    return clean
    