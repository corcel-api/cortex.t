from cortext.validating.managing.miner_manager import MinerManager
from cortext import CONFIG
from fastapi import FastAPI
from loguru import logger
import uvicorn
from pydantic import BaseModel
from typing import List
import asyncio
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create event loop if it doesn't exist
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Initialize MinerManager with the running loop
    global miner_manager
    miner_manager = MinerManager(
        network=CONFIG.subtensor_network,
        netuid=CONFIG.subtensor_netuid,
        wallet_name=CONFIG.wallet_name,
        wallet_hotkey=CONFIG.wallet_hotkey,
    )

    yield

    # Cleanup if needed
    await loop.shutdown_asyncgens()


app = FastAPI(lifespan=lifespan)


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


class TopPerformersRequest(BaseModel):
    n: int
    task_credit: int
    threshold: float = 1.0


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
    logger.info("Getting weights")
    uids, weights = miner_manager.weights
    return WeightsResponse(weights=weights, uids=uids)


@app.post("/api/consume_top_performers")
async def consume_top_performers(request: TopPerformersRequest):
    logger.info(
        f"Consuming {request.task_credit} credit for top {request.n} performers"
    )
    uids = miner_manager.consume_top_performers(
        n=request.n, task_credit=request.task_credit, threshold=request.threshold
    )
    return {"uids": uids}


if __name__ == "__main__":
    uvicorn.run(app, host=CONFIG.miner_manager.host, port=CONFIG.miner_manager.port)
