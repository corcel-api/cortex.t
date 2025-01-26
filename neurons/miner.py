from cortext import base, protocol, CONFIG, mining
from cortext.utilities.rate_limit import get_rate_limit_proportion
from cortext.validating.managing import ServingCounter
import bittensor as bt
import random
from typing import Tuple
import asyncio
import time
import httpx
from loguru import logger
import os
import redis
import traceback


class Miner(base.BaseMiner):
    def __init__(self):
        super().__init__([(self.forward_credit, self.blacklist_credit)])
        self.openai_client = httpx.AsyncClient(base_url="https://api.openai.com/v1")
        self.anthropic_client = httpx.AsyncClient(
            base_url="https://api.anthropic.com/v1"
        )
        self.redis = redis.Redis(host=CONFIG.redis.host, port=CONFIG.redis.port)
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

    def _initialize_rate_limits(self):
        r"""
        Initializes the rate limits for the miners.
        """

        S = self.metagraph.S
        valid_stake_uids = [
            uid for uid in range(len(S)) if S[uid] > CONFIG.bandwidth.min_stake
        ]
        rate_limit_distribution = {
            uid: max(
                int(
                    get_rate_limit_proportion(metagraph=self.metagraph, uid=uid)
                    * self.config.miner.total_credit
                ),
                2,
            )
            for uid in valid_stake_uids
        }
        self.rate_limits = {
            uid: ServingCounter(
                quota=rate_limit,
                uid=uid,
                redis_client=self.redis,
                postfix_key=self.axon.port,
            )
            for uid, rate_limit in rate_limit_distribution.items()
        }
        for uid, rate_limit in self.rate_limits.items():
            logger.info(f"Rate limit for {uid}: {rate_limit}")
        logger.info(f"Total credit: {self.config.miner.total_credit}")

    def run(self):
        bt.logging.info("Starting main loop")
        step = 0
        while True:
            try:
                # Periodically update our knowledge of the network graph.
                if step % 60 == 0:
                    self.metagraph.sync()
                    self._initialize_rate_limits()
                    log = (
                        f"Block: {self.metagraph.block.item()} | "
                        f"Incentive: {self.metagraph.I[self.uid]} | "
                    )
                    logger.info(log)
                step += 1
                time.sleep(10)

            except KeyboardInterrupt:
                self.axon.stop()
                bt.logging.success("Miner killed by keyboard interrupt.")
                break
            except Exception as e:
                bt.logging.error(f"Miner exception: {e}")
                bt.logging.error(traceback.format_exc())
                continue

    async def forward_credit(self, synapse: protocol.Credit) -> protocol.Credit:
        synapse.credit = self.config.miner.total_credit
        logger.info(f"Returning credit: {synapse.credit}")
        return synapse

    async def blacklist_credit(self, synapse: protocol.Credit) -> Tuple[bool, str]:
        return False, "Allowed"

    async def forward(self, synapse: protocol.ChatStreamingProtocol):
        payload = synapse.miner_payload
        logger.info(f"Payload: {payload}")
        if payload.model in ["gpt-4o-mini", "gpt-4o", "dall-e-3"]:
            response = await mining.forward.openai(
                self.openai_client, payload.model_dump()
            )
        elif payload.model in ["claude-3-5-sonnet-20241022"]:
            response = await mining.forward.claude(
                self.anthropic_client, payload.model_dump()
            )
        else:
            logger.error(f"Model {payload.model} not supported")
            return synapse
        logger.info(f"Response: {response}")

        async def stream_response(send):
            try:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        logger.debug(f"Streaming chunk: {line}")
                        await send(
                            {
                                "type": "http.response.body",
                                "body": (line + "\n\n").encode("utf-8"),
                                "more_body": True,
                            }
                        )
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error in stream_response: {e}")
                await send(
                    {
                        "type": "http.response.body",
                        "body": f'data: {{"error": "{str(e)}"}}\n\n'.encode("utf-8"),
                        "more_body": False,
                    }
                )

        return synapse.create_streaming_response(token_streamer=stream_response)

    def blacklist(self, synapse: protocol.ChatStreamingProtocol) -> Tuple[bool, str]:
        logger.info(f"Blacklisting {synapse}")
        hotkey = synapse.dendrite.hotkey
        uid = self.metagraph.hotkeys.index(hotkey)
        stake = self.metagraph.S[uid]
        if stake < CONFIG.bandwidth.min_stake:
            return True, "Stake too low."
        cost = CONFIG.bandwidth.model_configs[synapse.miner_payload.model].credit
        allowed = self.rate_limits[uid].increment(amount=cost)
        if not allowed:
            return True, "Rate limit exceeded."
        return False, ""


if __name__ == "__main__":
    miner = Miner()
    miner.run()
