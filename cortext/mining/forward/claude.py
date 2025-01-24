from httpx import AsyncClient
import os
from loguru import logger
from ...global_config import CONFIG
from ...protocol import mimic_chat_completion_chunk
import json
import uuid


class MockOpenAICompatibleResponse:
    def __init__(self, response, model: str = ""):
        self.response = response
        self.id = str(uuid.uuid4())
        self.model = model

    async def aiter_lines(self):
        for chunk in self.response.iter_text():
            if not chunk:
                continue

            for line in chunk.splitlines():
                if not line.strip() or not line.startswith("data: "):
                    continue

                try:
                    line_data = line.split("data: ")[1]
                    data = json.loads(line_data)

                    if data["type"] == "message_start":
                        self.id = data["message"]["id"]
                        self.model = data["message"]["model"]
                        continue

                    if data["type"] == "content_block_delta":
                        delta_text = data["delta"]["text"]
                        openai_data = mimic_chat_completion_chunk(
                            delta_text, id=self.id, model=self.model
                        )
                        yield f"data: {openai_data.model_dump_json()}\n\n"

                except Exception as e:
                    logger.error(f"Error processing line: {str(e)}")
                    continue


async def forward(client: AsyncClient, payload: dict):
    model = payload["model"]

    if model not in ["claude-3-5-sonnet-20241022"]:
        logger.error(f"Model {model} not supported in claude")
        return None

    allowed_params = CONFIG.bandwidth.model_configs[model].allowed_params
    valid_payload = {k: v for k, v in payload.items() if k in allowed_params}

    logger.info(f"Payload: {valid_payload}")

    response = await client.post(
        "/messages",
        json=valid_payload,
        headers={
            "x-api-key": f"{os.getenv('ANTHROPIC_API_KEY')}",
            "anthropic-version": "2023-06-01",
        },
        timeout=60.0,
    )

    return MockOpenAICompatibleResponse(response)
