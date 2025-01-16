from dotenv import load_dotenv

load_dotenv()

import bittensor as bt
from loguru import logger
from subnet_core import CONFIG
from redis import Redis
import json
import time
from tqdm import tqdm


redis_client = Redis(host=CONFIG.redis.host, port=CONFIG.redis.port, db=CONFIG.redis.db)
logger.info(f"Connected to Redis at {CONFIG.redis.host}:{CONFIG.redis.port}")


def create_synthetic_payload(model_name: str):
    if model_name in ["gpt-4o", "gpt-4o-mini"]:
        logger.debug(f"Creating synthetic payload for model {model_name}")
        return {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who are you?"},
            ],
        }
    else:
        logger.error(f"Model {model_name} not supported")
        raise ValueError(f"Model {model_name} not supported")


while True:
    logger.debug("Starting refill loop")
    models = CONFIG.bandwidth.model_configs.keys()
    for model in models:
        redis_key = f"{CONFIG.redis.synthetic_queue_key}:{model}"
        # Check pool size and fill until reach CONFIG.synthesize.synthetic_pool_size
        current_size = redis_client.llen(redis_key)
        logger.info(
            f"Current pool size for {model}: {current_size}/{CONFIG.synthesize.synthetic_pool_size}"
        )
        if current_size < CONFIG.synthesize.synthetic_pool_size:
            needed = CONFIG.synthesize.synthetic_pool_size - current_size
            logger.info(f"Refilling {needed} synthetic payloads for {model}")
            pbar = tqdm(range(needed), desc=f"Refilling {model} pool")
            for _ in pbar:
                payload = create_synthetic_payload(model)
                redis_client.rpush(redis_key, json.dumps(payload))
        else:
            logger.info(f"Pool for {model} is full")

    time.sleep(60)
