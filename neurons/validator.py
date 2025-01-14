from subnet_core import validating, CONFIG, base, protocol
import bittensor as bt
import random
import httpx
import asyncio
from loguru import logger
import traceback
import pandas as pd


class Validator(base.BaseValidator):
    def __init__(self):
        super().__init__()
        self.miner_manager = validating.managing.MinerManager(
            uid=self.uid, wallet=self.wallet, metagraph=self.metagraph
        )
        logger.info("Initialized miner manager")
        self.dendrite = bt.dendrite(wallet=self.wallet)
        logger.info("Initialized dendrite")
        self.score_client = httpx.AsyncClient(
            base_url=f"{CONFIG.score.host}:{CONFIG.score.port}"
        )
        logger.info(
            f"Initialized score client with base URL: {CONFIG.score.host}:{CONFIG.score.port}"
        )

    async def start_epoch(self):
        logger.info("Starting forward pass")
        self.miner_manager.sync_credit()
        

    async def load_streaming_response(self, response) -> protocol.ChatStreamingProtocol:
        async for chunk in response:
            continue
        return chunk

    async def score(
        self,
        uids: list[int],
        responses: list[protocol.ChatStreamingProtocol],
        base_request: protocol.ChatStreamingProtocol,
    ):
        logger.info(f"Starting scoring for {len(uids)} miners")
        score_response = await self.score_client.post(
            "/score",
            json={
                "responses": responses,
                "request": base_request,
            },
        )
        scores: list[float] = await score_response.json()
        logger.info(f"Updating miner manager with {len(scores)} scores")
        self.miner_manager.step(scores, uids)

    def set_weights(self):
        self.current_block = self.subtensor.get_current_block()
        self.last_update = self.metagraph.last_update[self.uid]
        weights = self.miner_manager.weights
        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        (
            uint_uids,
            uint_weights,
        ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        if self.current_block > self.last_update + CONFIG.subnet_tempo:
            weight_info = list(zip(uint_uids, uint_weights))
            weight_info_df = pd.DataFrame(weight_info, columns=["uid", "weight"])
            logger.info(f"Weight info:\n{weight_info_df.to_markdown()}")
            logger.info("Actually trying to set weights.")
            try:
                future = self.set_weights_executor.submit(
                    self.subtensor.set_weights,
                    netuid=self.config.netuid,
                    wallet=self.wallet,
                    uids=uint_uids,
                    weights=uint_weights,
                )
                success, msg = future.result(timeout=120)
                if not success:
                    logger.error(f"Failed to set weights: {msg}")
            except Exception as e:
                logger.error(f"Failed to set weights: {e}")
                traceback.print_exc()

            logger.info(f"Set weights result: {success}")
        else:
            logger.info(
                f"Not setting weights because current block {self.current_block} is not greater than last update {self.last_update} + tempo {constants.SUBNET_TEMPO}"
            )


if __name__ == "__main__":
    validator = Validator()
    asyncio.run(validator.run())
