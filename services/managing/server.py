from subnet_core.validating.managing.miner_manager import MinerManager
from subnet_core import CONFIG
from fastapi import FastAPI
from dotenv import load_dotenv
import os
from loguru import logger
import uvicorn
from pydantic import BaseModel
from typing import List

load_dotenv()

app = FastAPI()

class ConsumeRequest(BaseModel):
    threshold: float
    k: int 
    task_credit: int

class StepRequest(BaseModel):
    scores: List[float]
    total_uids: List[int]

class WeightsResponse(BaseModel):
    weights: List[float]
    uids: List[int]

miner_manager = MinerManager(
    network=os.getenv("SUBTENSOR.NETWORK"),
    netuid=os.getenv("NETUID"),
    wallet_name=os.getenv("WALLET.NAME"),
    wallet_hotkey=os.getenv("WALLET.HOTKEY"),
)

@app.post("/api/consume")
async def consume(request: ConsumeRequest):
    logger.info(f"Consuming {request.task_credit} credit for {request.k} miners")
    uids = miner_manager.consume(request.threshold, request.k, request.task_credit)
    return {"uids": uids}

@app.post("/api/step")
async def step(request: StepRequest):
    logger.info(f"Stepping {len(request.total_uids)} miners")
    try:
        miner_manager.step(request.scores, request.total_uids)
        return {"success": True}
    except Exception as e:
        logger.error(f"Error in step: {e}")
        return {"success": False}

@app.get("/api/weights")
async def weights():
    logger.info(f"Getting weights")
    uids, weights = miner_manager.weights
    return WeightsResponse(weights=weights, uids=uids)

if __name__ == "__main__":
    uvicorn.run(app, host=CONFIG.miner_manager.host, port=CONFIG.miner_manager.port)
