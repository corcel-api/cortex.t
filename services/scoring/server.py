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
    miner_responses = request.responses
    scores = []
    for response in miner_responses:
        scores.append(0.75)
    return ScoringResponse(scores=scores)
