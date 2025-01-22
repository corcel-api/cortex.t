from httpx import AsyncClient
import os
from ...protocol import ImagePrompt
import json
import asyncio
from typing import AsyncIterator
from loguru import logger


async def mock_stream_response(chunk: str) -> AsyncIterator[str]:
    # Yield the chunk with proper SSE format
    yield f"data: {chunk}"
    # Yield the [DONE] message
    yield "data: [DONE]"


async def forward(client: AsyncClient, payload: dict):
    model = payload["model"]
    if model in ["gpt-4o-mini", "gpt-4o"]:
        return await client.post(
            "/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
            timeout=60.0,
        )
    elif model in ["dall-e-3"]:
        image_prompt = payload["messages"][0]["content"]
        image_prompt = ImagePrompt.from_string(image_prompt)
        logger.info(f"image_prompt: {image_prompt}")
        response = await client.post(
            "/images/generations",
            json=image_prompt.model_dump(),
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
            timeout=60.0,
        )
        url = response.json()["data"][0]["url"]
        chunk = ImagePrompt.mimic_chat_completion_chunk(url)

        # Create a mock response object that implements aiter_lines
        class MockResponse:
            async def aiter_lines(self):
                async for line in mock_stream_response(chunk.model_dump_json()):
                    yield line

        return MockResponse()
