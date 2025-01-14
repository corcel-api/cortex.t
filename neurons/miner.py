from subnet_core import base, protocol, CONFIG
import bittensor as bt
import random
from typing import Tuple
import asyncio
import time


class Miner(base.BaseMiner):
    def __init__(self):
        super().__init__([(self.forward_credit, self.blacklist_credit)])

    async def forward_credit(self, synapse: protocol.Credit) -> protocol.Credit:
        synapse.credit = 256
        return synapse

    async def blacklist_credit(self, synapse: protocol.Credit) -> Tuple[bool, str]:
        return False, "Allowed"

    async def forward(self, synapse: protocol.ChatStreamingProtocol):
        async def yield_token():
            for _ in range(128):
                await asyncio.sleep(0.5)
                token = random.choice(["a", "b", "c", "d", "e", "f"])
                chunk = f"""data: {{"id":"chatcmpl-ApVyQCkzQJmZIrFIO0KCmVBr4rb16","object":"chat.completion.chunk","created":1736840998,"model":"gpt-4o-2024-08-06","service_tier":"default","system_fingerprint":"fp_50cad350e4","choices":[{{"index":0,"delta":{{"content":"{token}"}},"logprobs":null,"finish_reason":null}}]}}\n"""
                yield chunk

        tokens = yield_token()

        async def stream_response(send):
            i = 0
            async for token in tokens:
                i += 1
                if i > 16:
                    break
                await send(
                    {
                        "type": "http.response.body",
                        "body": token.encode("utf-8"),
                        "more_body": True,
                    }
                )
            await send(
                {
                    "type": "http.response.body",
                    "body": "[DONE]".encode("utf-8"),
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
