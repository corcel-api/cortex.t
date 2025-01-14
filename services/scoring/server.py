from fastapi import FastAPI
from subnet_core.protocol import ScoringRequest, ScoringResponse
import bittensor as bt

bt.logging.enable_default()
bt.logging.enable_info()
bt.logging.enable_debug()
bt.logging.enable_trace()

app = FastAPI()


@app.post("/score")
async def score(request: ScoringRequest):
    bt.logging.info(f"Scoring request received: {request}")
    return ScoringResponse(score=0.75)
