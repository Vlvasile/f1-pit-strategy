from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_plan_italian_gp():
    response = client.get("/plan", params={"event": "Italian Grand Prix", "race_laps": 53})
    assert response.status_code == 200
    assert response.json()["optimal_pit_lap"] == 25


def test_plan_unknown_event():
    response = client.get("/plan", params={"event": "Monza", "race_laps": 53})
    assert response.status_code == 404


def test_plan_race_too_short():
    response = client.get("/plan", params={"event": "Italian Grand Prix", "race_laps": 10})
    assert response.status_code == 422