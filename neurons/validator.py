from subnet_core import CONFIG, base, protocol
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
        self.score_client = httpx.AsyncClient(
            base_url=f"http://{CONFIG.score.host}:{CONFIG.score.port}"
        )
        self.synthesize_client = httpx.AsyncClient(
            base_url=f"http://{CONFIG.synthesize.host}:{CONFIG.synthesize.port}"
        )
        self.miner_manager_client = httpx.AsyncClient(
            base_url=f"http://{CONFIG.miner_manager.host}:{CONFIG.miner_manager.port}"
        )
        self.w_subtensor_client = httpx.AsyncClient(
            base_url=f"http://{CONFIG.w_subtensor.host}:{CONFIG.w_subtensor.port}"
        )

    async def start_epoch(self):
        batch_size = CONFIG.validating.synthetic_batch_size
        concurrent_batches = CONFIG.validating.synthetic_concurrent_batches
        synthetic_threshold = CONFIG.validating.synthetic_threshold
        logger.info(
            f"Starting forward pass - {batch_size} batch size, {concurrent_batches} concurrent batches"
        )
        futures = []
        for _ in range(concurrent_batches):
            model_config = CONFIG.bandwidth.sample_model
            response = await self.miner_manager_client.post(
                "/api/consume",
                json={
                    "threshold": synthetic_threshold,
                    "k": batch_size,
                    "task_credit": model_config.credit,
                },
            )
            response_json = response.json()
            uids = response_json["uids"]
            logger.info(f"Synthesizing {uids} miners")
            synapse = await self.synthesize(model_config)
            futures.append(self.process_batch(uids, synapse, model_config))
        await asyncio.gather(*futures)
        await asyncio.sleep(60)
        logger.info("Finished forward pass")

    async def process_batch(self, uids, synapse, model_config):
        axons_data = await self.w_subtensor_client.post("/api/axons", json=uids)
        axons_data: list[str] = axons_data.json()
        axons = [bt.AxonInfo.from_string(axon_data) for axon_data in axons_data]
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
            json=model_config.model_dump(),
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
        result = await self.miner_manager_client.post(
            "/api/step",
            json={
                "scores": [0.0] * len(invalid_responses),
                "total_uids": invalid_uids,
            },
        )
        result_json = result.json()
        if result_json["success"]:
            logger.success("Stepped miners")
        else:
            logger.error("Failed to step miners")
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
            await self.miner_manager_client.post(
                "/api/step",
                json={
                    "scores": scores,
                    "total_uids": valid_uids,
                },
            )


if __name__ == "__main__":
    validator = Validator()
    asyncio.run(validator.run())
