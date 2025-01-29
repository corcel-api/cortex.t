from fastapi import FastAPI
from cortext.configs.bandwidth import ModelConfig
from cortext.protocol import MinerPayload
from cortext import CONFIG
import bittensor as bt
import uvicorn
from redis.asyncio import Redis
import json

bt.logging.enable_default()
bt.logging.enable_info()
bt.logging.enable_debug()
bt.logging.enable_trace()

app = FastAPI()

redis_client = Redis(host=CONFIG.redis.host, port=CONFIG.redis.port, db=CONFIG.redis.db)


@app.post("/synthesize")
async def synthesize(model_config: ModelConfig):
    bt.logging.info(f"Synthesizing request received: {model_config}")
    redis_synthetic_key = f"{CONFIG.redis.synthetic_queue_key}:{model_config.model}"
    redis_organic_key = f"{CONFIG.redis.organic_queue_key}:{model_config.model}"
    payload = await redis_client.lpop(redis_organic_key)
    if not payload:
        payload = await redis_client.lpop(redis_synthetic_key)
    payload = json.loads(payload)
    return {
        "miner_payload": MinerPayload(**payload),
    }


if __name__ == "__main__":
    uvicorn.run(app, host=CONFIG.synthesize.host, port=CONFIG.synthesize.port)
