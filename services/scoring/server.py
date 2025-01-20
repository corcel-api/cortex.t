from fastapi import FastAPI
from cortext.protocol import ScoringRequest, ScoringResponse
from cortext import CONFIG
import bittensor as bt
from openai import AsyncOpenAI
import numpy as np
import uvicorn
from loguru import logger

bt.logging.enable_default()
bt.logging.enable_info()
bt.logging.enable_debug()
bt.logging.enable_trace()

app = FastAPI()

CLIENT = AsyncOpenAI()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@app.post("/score")
async def score(request: ScoringRequest):
    bt.logging.info(f"Scoring request received: {request}")
    miner_completions: list[str] = request.responses

    # DOUBLE SPEND
    payload = request.request.model_dump()
    payload["stream"] = False
    output = await CLIENT.chat.completions.create(**payload)
    reference_completion = output.choices[0].message.content

    logger.info(f"Reference completion: {reference_completion}")

    texts = [reference_completion] + miner_completions
    output = await CLIENT.embeddings.create(input=texts, model="text-embedding-3-small")
    logger.info("Received embeddings")
    embeddings = [o.embedding for o in output.data]
    ref_embedding = embeddings[0]
    miner_embeddings = embeddings[1:]
    scores = [
        cosine_similarity(ref_embedding, miner_embedding)
        for miner_embedding in miner_embeddings
    ]
    logger.info(f"Scores: {scores}")
    return ScoringResponse(scores=scores)


if __name__ == "__main__":
    uvicorn.run(app, host=CONFIG.score.host, port=CONFIG.score.port)
