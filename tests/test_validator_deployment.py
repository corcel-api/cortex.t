import pytest
from cortext import CONFIG
import httpx


@pytest.fixture(scope="module")
def subtensor_client():
    with httpx.Client(
        base_url=f"http://{CONFIG.w_subtensor.host}:{CONFIG.w_subtensor.port}"
    ) as client:
        yield client


@pytest.fixture(scope="module")
def miner_manager_client():
    with httpx.Client(
        base_url=f"http://{CONFIG.miner_manager.host}:{CONFIG.miner_manager.port}"
    ) as client:
        yield client


@pytest.fixture(scope="module")
def synthesizing_client():
    with httpx.Client(
        base_url=f"http://{CONFIG.synthesize.host}:{CONFIG.synthesize.port}"
    ) as client:
        yield client


def test_subtensor_client(subtensor_client):
    response = subtensor_client.post("/api/axons", json={"uids": [1, 2]})
    assert response.status_code == 200
    print(response.json())


def test_miner_manager_client(miner_manager_client):
    response = miner_manager_client.get("/api/weights")
    assert response.status_code == 200
    print(response.json())


def test_synthesizing_client(synthesizing_client):
    model_config = CONFIG.bandwidth.sample_model
    response = synthesizing_client.post("/synthesize", json=model_config.model_dump())
    assert response.status_code == 200
    print(response.json())
