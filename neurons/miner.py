from subnet_core import base, protocol, CONFIG
import bittensor as bt
import random
from typing import Tuple
import asyncio
import time
from openai import AsyncOpenAI
from loguru import logger


class Miner(base.BaseMiner):
    def __init__(self):
        super().__init__([(self.forward_credit, self.blacklist_credit)])
        self.client = AsyncOpenAI()

    async def forward_credit(self, synapse: protocol.Credit) -> protocol.Credit:
        synapse.credit = 256
        return synapse

    async def blacklist_credit(self, synapse: protocol.Credit) -> Tuple[bool, str]:
        return False, "Allowed"

    async def forward(self, synapse: protocol.ChatStreamingProtocol):
        payload = synapse.miner_payload
        logger.info(f"Payload: {payload}")
        response = await self.client.chat.completions.create(
            **payload.model_dump(),
        )
        logger.info(f"Response: {response}")

        async def stream_response(send):
            try:
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        logger.debug(
                            f"Streaming chunk: {chunk.choices[0].delta.content}"
                        )
                        await send(
                            {
                                "type": "http.response.body",
                                "body": f"data: {chunk.model_dump_json()}\n\n".encode(
                                    "utf-8"
                                ),
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
