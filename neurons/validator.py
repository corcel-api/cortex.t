from subnet_core import validating, CONFIG, base, protocol
from subnet_core.configs.bandwidth import ModelConfig
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
            base_url=f"http://{CONFIG.score.host}:{CONFIG.score.port}"
        )
        self.synthesize_client = httpx.AsyncClient(
            base_url=f"http://{CONFIG.synthesize.host}:{CONFIG.synthesize.port}"
        )
        logger.info(
            f"Initialized score client with base URL: http://{CONFIG.score.host}:{CONFIG.score.port}"
        )

    async def start_epoch(self):
        logger.info("Starting forward pass")
        batch_size = 4
        concurrent_batches = 4
        futures = []
        for _ in range(concurrent_batches):
            model_config = CONFIG.bandwidth.sample_model
            uids = self.miner_manager.consume(
                threshold=0.5, k=batch_size, task_credit=model_config.credit
            )
            synapse = await self.synthesize(model_config)
            futures.append(self.process_batch(uids, synapse, model_config))
        await asyncio.gather(*futures)

    async def process_batch(self, uids, synapse, model_config):
        axons = [self.metagraph.axons[uid] for uid in uids]
        responses = await self.query_non_streaming(axons, synapse, model_config)
        await self.score(uids, responses, synapse)

    async def query_non_streaming(self, axons, synapse, model_config):
        if model_config.synapse_type == "streaming-chat":
            responses = await self.dendrite.forward(
                axons=axons,
                synapse=synapse,
                timeout=model_config.timeout,
                streaming=True,
            )
            # process responses by load_streaming_response
            futures = [self.load_streaming_response(response) for response in responses]
            responses = await asyncio.gather(*futures)
            return responses
        else:
            raise ValueError(f"Invalid synapse type: {model_config.synapse_type}")

    async def synthesize(self, model_config: ModelConfig):
        response = await self.synthesize_client.post(
            "/synthesize",
            json={
                "model_config": model_config.model_dump(),
            },
        )
        response_json = response.json()
        if model_config.synapse_type == "streaming-chat":
            synapse_cls = protocol.ChatStreamingProtocol
        else:
            raise ValueError(f"Invalid synapse type: {model_config.synapse_type}")

        synapse = synapse_cls(**response_json)

        return synapse

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
        valid_responses = []
        valid_uids = []
        invalid_responses = []
        invalid_uids = []
        for uid, response in zip(uids, responses):
            if not response or not response.is_success or not response.verify():
                invalid_responses.append(response)
                invalid_uids.append(uid)
                logger.error(f"Invalid response from miner {uid}")
            else:
                valid_responses.append(response)
                valid_uids.append(uid)
                logger.success(f"Valid response from miner {uid}")
        logger.info(f"Zeroing out scores for {len(invalid_responses)} miners")
        self.miner_manager.step(
            scores=[0.0] * len(invalid_responses), total_uids=invalid_uids
        )
        if len(valid_responses) == 0:
            logger.error("No valid responses found")
            return
        else:
            logger.info(f"Starting scoring for {len(valid_responses)} miners")
            score_response = await self.score_client.post(
                "/score",
                json={
                    "responses": [r.miner_response for r in valid_responses],
                    "request": base_request.miner_payload.model_dump(),
                },
            )
            scores: list[float] = score_response.json()["scores"]
            logger.info(f"Updating miner manager with {len(scores)} scores")
            self.miner_manager.step(scores, valid_uids)

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
