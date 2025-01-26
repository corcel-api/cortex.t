import numpy as np
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from cortext import CONFIG
import os

OPENAI_CLIENT = AsyncOpenAI()
ANTHROPIC_CLIENT = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


async def create_ground_truth(payload: dict) -> str:
    """Generate ground truth completion from either Claude or OpenAI."""
    payload["stream"] = False

    if "claude" in payload["model"]:
        payload = {
            k: v
            for k, v in payload.items()
            if k in CONFIG.bandwidth.model_configs[payload["model"]].allowed_params
        }
        output = await ANTHROPIC_CLIENT.messages.create(**payload)
        return output.content[0].text
    elif "gpt" in payload["model"]:
        output = await OPENAI_CLIENT.chat.completions.create(**payload)
        return output.choices[0].message.content
    else:
        raise ValueError(f"Unsupported model: {payload['model']}")


async def create_embeddings(payload: dict) -> np.ndarray:
    """Create embeddings for a given text."""
    output = await OPENAI_CLIENT.embeddings.create(**payload)
    return [d.embedding for d in output.data]
