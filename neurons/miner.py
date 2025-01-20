from cortext import base, protocol, CONFIG
import bittensor as bt
import random
from typing import Tuple
import asyncio
import time
import httpx
from loguru import logger
import os


class Miner(base.BaseMiner):
    def __init__(self):
        super().__init__([(self.forward_credit, self.blacklist_credit)])
        self.client = httpx.AsyncClient(base_url="https://api.openai.com/v1")

    async def forward_credit(self, synapse: protocol.Credit) -> protocol.Credit:
        synapse.credit = 256
        return synapse

    async def blacklist_credit(self, synapse: protocol.Credit) -> Tuple[bool, str]:
        return False, "Allowed"

    async def forward(self, synapse: protocol.ChatStreamingProtocol):
        payload = synapse.miner_payload
        logger.info(f"Payload: {payload}")
        response = await self.client.post(
            "/chat/completions",
            json=payload.model_dump(),
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
            timeout=60.0,
        )
        logger.info(f"Response: {response}")

        async def stream_response(send):
            try:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        logger.debug(f"Streaming chunk: {line}")
                        await send(
                            {
                                "type": "http.response.body",
                                "body": f"{line}\n".encode("utf-8"),
                                "more_body": True,
                            }
                        )
                # Send the final [DONE] message
                await send(
                    {
                        "type": "http.response.body",
                        "body": "data: [DONE]\n\n".encode("utf-8"),
                        "more_body": False,
                    }
                )
            except Exception as e:
                logger.error(f"Error in stream_response: {e}")
                # Send error message and close the stream
                await send(
                    {
                        "type": "http.response.body",
                        "body": f'data: {{"error": "{str(e)}"}}\n\n'.encode("utf-8"),
                        "more_body": False,
                    }
                )

        return synapse.create_streaming_response(token_streamer=stream_response)

    async def blacklist(
        self, synapse: protocol.ChatStreamingProtocol
    ) -> Tuple[bool, str]:
        return False, "Allowed"


if __name__ == "__main__":
    import time

    miner = Miner()
    while True:
        miner.chain_sync()
        print("Synced")
        time.sleep(60)
