from httpx import AsyncClient
import os
from ...protocol import ImagePrompt, mimic_chat_completion_chunk
from loguru import logger


async def forward(client: AsyncClient, payload: dict):
    model = payload["model"]
    if model in ["gpt-4o-mini", "gpt-4o"]:
        # Create a mock response object similar to the DALL-E implementation
        class MockResponse:
            def __init__(self, stream_response):
                self.stream_response = stream_response

            async def aiter_lines(self):
                async with self.stream_response as response:
                    async for line in response.aiter_lines():
                        yield line

        stream_response = client.stream(
            method="post",
            url="/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
            timeout=60.0,
        )
        return MockResponse(stream_response)
    elif model in ["dall-e-3"]:
        # Extract the prompt from the messages
        image_prompt = payload["messages"][0]["content"]
        image_prompt = ImagePrompt.from_string(image_prompt)
        logger.info(f"image_prompt: {image_prompt}")

        # Prepare the DALL-E specific payload
        dalle_payload = {
            "model": "dall-e-3",
            "prompt": image_prompt.prompt,
            "n": 1,
            "size": image_prompt.size or "1024x1024",  # Default size if not specified
            "quality": image_prompt.quality
            or "standard",  # Default quality if not specified
            "style": image_prompt.style or "vivid",  # Default style if not specified
        }

        try:
            future_response = client.post(
                "/images/generations",
                json=dalle_payload,
                headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
                timeout=60.0,
            )

            # Create a mock response object that implements aiter_lines
            class MockResponse:
                async def aiter_lines(self):
                    response = await future_response
                    if response.status_code != 200:
                        logger.error(f"DALL-E API error: {response.text}")
                        raise Exception(f"DALL-E API error: {response.text}")

                    response_data = response.json()
                    url = response_data["data"][0]["url"]
                    chunk = mimic_chat_completion_chunk(url)
                    logger.info(f"chunk: {chunk}")
                    yield f"data: {chunk.model_dump_json()}"
                    yield "data: [DONE]"

            return MockResponse()

        except Exception as e:
            logger.error(f"Error in DALL-E processing: {str(e)}")
            raise
    else:
        logger.error(f"Model {model} not supported in openai")
        return None
