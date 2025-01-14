from fastapi import FastAPI
from subnet_core.configs.bandwidth import ModelConfig
from subnet_core.protocol import MinerPayload
import bittensor as bt
import random

bt.logging.enable_default()
bt.logging.enable_info()
bt.logging.enable_debug()
bt.logging.enable_trace()

app = FastAPI()


@app.post("/synthesize")
async def synthesize(model_config: ModelConfig):
    bt.logging.info(f"Synthesizing request received: {model_config}")
    return {
        "miner_payload": MinerPayload(
            model=model_config.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can synthesize text.",
                }
            ],
            max_tokens=model_config.max_tokens,
            temperature=round(random.uniform(0.5, 1.0), 2),
        ),
    }
