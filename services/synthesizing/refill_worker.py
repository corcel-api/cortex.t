from dotenv import load_dotenv

load_dotenv()

import bittensor as bt
from loguru import logger
from cortext import CONFIG
from redis import Redis
import json
import time
from tqdm import tqdm
from datasets import load_dataset
import random
from cortext.protocol import MinerPayload, ImagePrompt
import random

ds = load_dataset(
    "HuggingFaceFW/fineweb-edu", "sample-100BT", streaming=True, split="train"
)
ds = ds.filter(lambda x: len(x["text"]) < 3000)
ds = iter(ds)
redis_client = Redis(host=CONFIG.redis.host, port=CONFIG.redis.port, db=CONFIG.redis.db)
logger.info(f"Connected to Redis at {CONFIG.redis.host}:{CONFIG.redis.port}")

# Clear existing queues on startup
for model in CONFIG.bandwidth.model_configs.keys():
    redis_key = f"{CONFIG.redis.synthetic_queue_key}:{model}"
    redis_client.delete(redis_key)
    logger.info(f"Cleared existing queue for {model}")


def create_synthetic_payload(model_name: str):
    if model_name in ["gpt-4o", "claude-3-5-sonnet-20241022"]:
        n_turn = random.randint(1, 2)
        messages = []
        for i in range(n_turn):
            text = next(ds)["text"]
            text_length = len(text)
            user_length = int(text_length * 0.4)
            user_content = text[:user_length]
            assistant_content = text[user_length:]
            messages.append({"role": "user", "content": user_content})
            if i < n_turn - 1:
                messages.append({"role": "assistant", "content": assistant_content})
        return {
            "model": model_name,
            "messages": messages,
            "seed": random.randint(0, 1000000),
            "temperature": round(random.random() * 0.4, 2),
        }
    elif model_name in ["dall-e-3"]:
        text = next(ds)["text"]
        sentences = text.split(".")
        sentences = [s for s in sentences if len(s) > 2]
        caption = random.choice(sentences)
        size = random.choice(["1024x1024", "1792x1024", "1024x1792"])
        style = random.choice(["natural", "vivid"])
        quality = random.choice(["standard", "hd"])
        prompt = ImagePrompt(
            prompt=caption,
            size=size,
            style=style,
            quality=quality,
        )
        prompt = prompt.to_string()
        return {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
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
                payload = MinerPayload(
                    **payload,
                )
                redis_client.rpush(redis_key, payload.model_dump_json())
        else:
            logger.info(f"Pool for {model} is full")

    time.sleep(60)
