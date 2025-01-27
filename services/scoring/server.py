from fastapi import FastAPI
from cortext.protocol import ScoringRequest, ScoringResponse, ImagePrompt
from cortext import CONFIG
import uvicorn
from loguru import logger
from .scorers.text import create_ground_truth, cosine_similarity, create_embeddings
from .scorers.image import dall_e_deterministic_score

app = FastAPI()


@app.post("/score")
async def score(request: ScoringRequest) -> ScoringResponse:
    """Score miner responses against ground truth using embeddings."""
    logger.info(f"Scoring request received: {request}")
    miner_completions: list[str] = request.responses
    payload = request.request.model_dump()
    model = request.request.model

    scores = []
    if model in ["dall-e-3"]:
        prompt = request.request.messages[0]["content"]
        prompt = ImagePrompt.from_string(prompt)
        for image_url in miner_completions:
            score = dall_e_deterministic_score(
                image_url=image_url, prompt=prompt.prompt, size=prompt.size
            )
            scores.append(score)
    elif model in ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet-20241022"]:
        payload["stream"] = False
        reference_completion = await create_ground_truth(payload)
        texts = [reference_completion] + miner_completions
        logger.info(f"Miner completions: {miner_completions}")
        logger.info(f"Reference completion: {reference_completion}")
        logger.info(f"Creating embeddings for {len(texts)} texts")
        embeddings = await create_embeddings(
            {
                "input": texts,
                "model": "text-embedding-3-large",
            }
        )
        logger.info(f"Received {len(embeddings)} embeddings")
        ref_embedding = embeddings[0]
        miner_embeddings = embeddings[1:]
        scores = [
            cosine_similarity(ref_embedding, miner_embedding)
            for miner_embedding in miner_embeddings
        ]
    else:
        raise ValueError(f"Unsupported model: {model}")

    logger.info(f"{model}|{scores}")

    return ScoringResponse(scores=scores)


if __name__ == "__main__":
    uvicorn.run(app, host=CONFIG.score.host, port=CONFIG.score.port)
