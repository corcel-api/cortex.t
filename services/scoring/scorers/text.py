import numpy as np
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

OPENAI_CLIENT = AsyncOpenAI()
ANTHROPIC_CLIENT = AsyncAnthropic()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


async def create_ground_truth(payload: dict) -> str:
    """Generate ground truth completion from either Claude or OpenAI."""
    payload["stream"] = False

    if "claude" in payload["model"]:
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
    return output.data[0].embedding
