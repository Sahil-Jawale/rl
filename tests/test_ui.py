from __future__ import annotations

from fastapi.testclient import TestClient

from chess_endgame_rl.ui.server import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_scenarios_endpoint():
    response = client.get('/api/scenarios')
    assert response.status_code == 200
    data = response.json()
    assert 'scenarios' in data
    assert 'krk' in data['scenarios']


def test_results_endpoint_returns_json_list():
    response = client.get('/api/results')
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
