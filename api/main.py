from fastapi import FastAPI, HTTPException

from src.strategy import load_pit_loss, plan_one_stop

app = FastAPI(title="F1 Pit Strategy API")

PIT_LOSS = load_pit_loss()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/events")
def events() -> list[str]:
    return sorted(PIT_LOSS)


@app.get("/plan")
def plan(event: str, race_laps: int):
    try:
        return plan_one_stop(event, race_laps, PIT_LOSS)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown event: {event}. See /events for valid names.")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))