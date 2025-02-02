from cortext import CONFIG, base, protocol
from cortext.configs.bandwidth import ModelConfig
import bittensor as bt
import httpx
import asyncio
from loguru import logger
import traceback
import time
import numpy as np
from redis.asyncio import Redis


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
        self.redis = Redis(host="localhost", port=6379, db=1)

    async def periodically_set_weights(self):
        while not self.should_exit:
            scored_counter = await self.get_scored_counter()
            logger.info(
                f"Total scored: {len(await self.redis.keys('scored_uid:*'))} in this epoch"
            )
            scored_times = np.array(list(scored_counter.values()))
            mean = np.mean(scored_times) if len(scored_times) > 0 else 0
            std = np.std(scored_times) if len(scored_times) > 0 else 0
            logger.info(f"Mean scored times per uid: {mean} ± {std}")
            result = await self.w_subtensor_client.post("/api/set_weights", timeout=120)
            result = result.json()
            logger.info(f"Set weights result: {result}")
            if result["success"]:
                logger.info("Resetting scored_uids after setting weights successfully")
                await self.redis.flushdb()
            await asyncio.sleep(600)

    async def run(self):
        logger.info("Starting validator loop.")
        asyncio.create_task(self.periodically_set_weights())
        logger.info("Resetting scored_uids")
        await self.redis.flushdb()
        while not self.should_exit:
            try:
                await self.start_epoch()
            except Exception as e:
                logger.error(f"Forward error: {e}")
                traceback.print_exc()
            except KeyboardInterrupt:
                logger.success("Validator killed by keyboard interrupt.")
                exit()

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
            try:
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
                if not uids:
                    logger.error("No miners found")
                    continue
                logger.info(f"Forwarding - {uids} - {model_config.model}")
                synapse = await self.synthesize(model_config)
                futures.append(self.process_batch(uids, synapse, model_config))
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in start_epoch: {str(e)}")
                continue

        await asyncio.gather(*futures)
        await asyncio.sleep(4)

    async def process_batch(self, uids, synapse, model_config):
        try:
            axons_data = await self.w_subtensor_client.get(
                "/api/axons", timeout=4, params={"uids": uids}
            )
            axons_data: list[str] = axons_data.json()
            axons = [bt.AxonInfo.from_string(axon_data) for axon_data in axons_data]
            responses = await self.query_non_streaming(axons, synapse, model_config)
            logger.info(f"Received {len(responses)} responses")
            await self.score(uids, responses, synapse)
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in process_batch: {str(e)}")

    async def query_non_streaming(self, axons, synapse, model_config):
        if model_config.synapse_type == "streaming-chat":
            try:
                responses = await self.dendrite.forward(
                    axons=axons,
                    synapse=synapse,
                    timeout=model_config.timeout,
                    streaming=True,
                )
                futures = [
                    self.load_streaming_response(response) for response in responses
                ]
                responses = await asyncio.gather(*futures)
                return responses
            except Exception as e:
                logger.error(f"Error in query_non_streaming: {str(e)}")
                return []
        else:
            raise ValueError(f"Invalid synapse type: {model_config.synapse_type}")

    async def synthesize(self, model_config: ModelConfig):
        try:
            response = await self.synthesize_client.post(
                "/synthesize",
                json=model_config.model_dump(),
                timeout=16,
            )
            response_json = response.json()
            if model_config.synapse_type == "streaming-chat":
                synapse_cls = protocol.ChatStreamingProtocol
            else:
                raise ValueError(f"Invalid synapse type: {model_config.synapse_type}")

            synapse = synapse_cls(**response_json)
            return synapse
        except Exception as e:
            logger.error(f"Error in synthesize: {str(e)}")
            raise

    async def load_streaming_response(self, response) -> protocol.ChatStreamingProtocol:
        try:
            start_time = time.time()
            async for chunk in response:
                logger.info(f"Loading Streaming Response - {time.time() - start_time}s")
                continue
            end_time = time.time()
            chunk.dendrite.process_time = end_time - start_time
            return chunk
        except Exception as e:
            logger.error(f"Error in load_streaming_response: {str(e)}")
            return None, 0

    async def get_scored_counter(self):
        scored_counter = {}
        async for key in self.redis.scan_iter("scored_uid:*"):
            uid = int(key.decode().split(":")[1])
            count = await self.redis.get(key)
            scored_counter[uid] = int(count)
        return scored_counter

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
                logger.info(
                    f"Invalid response - {uid} - {response.dendrite.process_time}s"
                )
                logger.info(f"Response: {response}")
            else:
                valid_responses.append(response)
                valid_uids.append(uid)
                logger.info(
                    f"Valid response - {uid} - {response.dendrite.process_time}s"
                )
        logger.info(f"Valid UIDs: {valid_uids}")

        uid_counts = await self.get_scored_counter()

        valid_uids_to_score = []
        for uid in valid_uids:
            if uid_counts.get(uid, 0) < 4:
                valid_uids_to_score.append(uid)
            else:
                logger.info(f"Ignoring UID: {uid} - {uid_counts.get(uid, 0)}")
        logger.info(
            f"Valid UIDs to score (after frequency filter): {valid_uids_to_score}"
        )

        if invalid_uids:
            try:
                result = await self.miner_manager_client.post(
                    "/api/step",
                    json={
                        "scores": [0.0] * len(invalid_responses),
                        "total_uids": invalid_uids,
                    },
                    timeout=60.0,
                )
                result.raise_for_status()
            except Exception as e:
                logger.error(f"Error zeroing invalid miners: {str(e)}")

        if not valid_uids_to_score:
            return

        try:
            score_response = await self.score_client.post(
                "/score",
                json={
                    "responses": [r.miner_response for r in valid_responses],
                    "request": base_request.miner_payload.model_dump(),
                },
                timeout=60.0,
            )
            scores: list[float] = score_response.json()["scores"]
            logger.info(f"Scores: {scores}")
            await self.miner_manager_client.post(
                "/api/step",
                json={
                    "scores": scores,
                    "total_uids": valid_uids_to_score,
                },
            )
            pipe = self.redis.pipeline()
            for uid in valid_uids_to_score:
                key = f"scored_uid:{uid}"
                pipe.incr(key)
                pipe.expire(key, 360)
            await pipe.execute()
        except Exception as e:
            logger.error(f"Error in scoring valid responses: {str(e)}")


if __name__ == "__main__":
    validator = Validator()
    asyncio.run(validator.run())
