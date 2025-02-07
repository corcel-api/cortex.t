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
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ClientConfig:
    """Configuration for HTTP clients"""

    score_url: str = f"http://{CONFIG.score.host}:{CONFIG.score.port}"
    synthesize_url: str = f"http://{CONFIG.synthesize.host}:{CONFIG.synthesize.port}"
    miner_manager_url: str = (
        f"http://{CONFIG.miner_manager.host}:{CONFIG.miner_manager.port}"
    )
    w_subtensor_url: str = f"http://{CONFIG.w_subtensor.host}:{CONFIG.w_subtensor.port}"


@dataclass
class ResponseTrackingData:
    """Data structure for tracking response metrics"""

    batch_id: str
    uid: int
    model: str
    score: float
    response_time: float
    invalid_reason: str = ""
    timestamp: float = field(default_factory=time.time)


class ResponseProcessor:
    """Handles processing and validation of miner responses"""

    @staticmethod
    def validate_responses(
        uids: List[int], responses: List[protocol.ChatStreamingProtocol]
    ) -> Tuple[
        List[Tuple[int, protocol.ChatStreamingProtocol]],
        List[Tuple[int, protocol.ChatStreamingProtocol, str]],
    ]:
        valid = []
        invalid = []

        for uid, response in zip(uids, responses):
            if response and response.is_success and response.verify():
                valid.append((uid, response))
                logger.info(
                    f"Valid response - {uid} - {response.dendrite.process_time}s"
                )
            else:
                invalid_reason = ""
                if not response:
                    invalid_reason = "no_response"
                elif not response.is_success:
                    invalid_reason = "not_successful"
                elif not response.verify():
                    invalid_reason = "verification_failed"

                invalid.append((uid, response, invalid_reason))
                logger.info(
                    f"Invalid response - {uid} - {response.dendrite.process_time if response else 0}s - Reason: {invalid_reason}"
                )

        return valid, invalid


class Validator(base.BaseValidator):
    def __init__(self):
        super().__init__()
        self._init_clients(ClientConfig())
        self.redis = Redis(host="localhost", port=6379, db=1)
        self.response_processor = ResponseProcessor()

    def _init_clients(self, config: ClientConfig) -> None:
        """Initialize HTTP clients"""
        self.score_client = httpx.AsyncClient(base_url=config.score_url)
        self.synthesize_client = httpx.AsyncClient(base_url=config.synthesize_url)
        self.miner_manager_client = httpx.AsyncClient(base_url=config.miner_manager_url)
        self.w_subtensor_client = httpx.AsyncClient(base_url=config.w_subtensor_url)

    async def run(self) -> None:
        """Main validator loop"""
        logger.info("Starting validator loop.")
        asyncio.create_task(self.periodically_set_weights())
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

    async def periodically_set_weights(self) -> None:
        """Periodically update network weights based on scoring"""
        while not self.should_exit:
            try:
                await self._update_weights()
                await asyncio.sleep(600)
            except Exception as e:
                logger.error(f"Error updating weights: {e}")
                await asyncio.sleep(60)  # Shorter sleep on error

    async def _update_weights(self) -> None:
        """Update network weights and handle scoring data"""
        scored_counter = await self.get_scored_counter()
        await self._log_scoring_statistics(scored_counter)

        result = await self.w_subtensor_client.post("/api/set_weights", timeout=120)
        result = result.json()
        logger.info(f"Set weights result: {result}")

        if result["success"]:
            logger.info("Resetting scored_uids after setting weights successfully")
            await self.redis.flushdb()

    async def _log_scoring_statistics(self, scored_counter: Dict[int, int]) -> None:
        """Log statistics about scoring"""
        logger.info(
            f"Total scored: {len(await self.redis.keys('scored_uid:*'))} in this epoch"
        )
        scored_times = np.array(list(scored_counter.values()))

        if len(scored_times) > 0:
            mean = np.mean(scored_times)
            std = np.std(scored_times)
            logger.info(f"Mean scored times per uid: {mean} ± {std}")

    async def start_epoch(self) -> None:
        """Start a new validation epoch"""
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
                uids = await self._get_miner_uids(
                    model_config, batch_size, synthetic_threshold
                )
                if not uids:
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

    async def _get_miner_uids(
        self, model_config: ModelConfig, batch_size: int, threshold: float
    ) -> Optional[List[int]]:
        """Get UIDs of available miners"""
        response = await self.miner_manager_client.post(
            "/api/consume",
            json={
                "threshold": threshold,
                "k": batch_size,
                "task_credit": model_config.credit,
            },
        )
        response_json = response.json()
        uids = response_json["uids"]

        if not uids:
            logger.error("No miners found")
        return uids

    async def process_batch(
        self,
        uids: List[int],
        synapse: protocol.ChatStreamingProtocol,
        model_config: ModelConfig,
    ) -> None:
        """Process a batch of miners"""
        batch_id = f"batch_{int(time.time())}_{model_config.model}"
        try:
            axons = await self._get_axons(uids)
            responses = await self.query_non_streaming(axons, synapse, model_config)
            logger.info(f"Received {len(responses)} responses")
            await self.score(uids, responses, synapse, batch_id)
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in process_batch: {str(e)}")

    async def _get_axons(self, uids: List[int]) -> List[bt.AxonInfo]:
        """Get axon information for UIDs"""
        axons_data = await self.w_subtensor_client.post(
            "/api/axons",
            json={"uids": uids},
            timeout=4,
        )
        axons_data: List[str] = axons_data.json()["axons"]
        return [bt.AxonInfo.from_string(axon_data) for axon_data in axons_data]

    async def query_non_streaming(
        self,
        axons: List[bt.AxonInfo],
        synapse: protocol.ChatStreamingProtocol,
        model_config: ModelConfig,
    ) -> List[protocol.ChatStreamingProtocol]:
        """Query miners with non-streaming protocol"""
        if model_config.synapse_type != "streaming-chat":
            raise ValueError(f"Invalid synapse type: {model_config.synapse_type}")

        try:
            responses = await self.dendrite.forward(
                axons=axons,
                synapse=synapse,
                timeout=model_config.timeout,
                streaming=True,
            )
            futures = [self.load_streaming_response(response) for response in responses]
            return await asyncio.gather(*futures)
        except Exception as e:
            logger.error(f"Error in query_non_streaming: {str(e)}")
            return []

    async def synthesize(
        self, model_config: ModelConfig
    ) -> protocol.ChatStreamingProtocol:
        """Create a synapse from model config"""
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

            return synapse_cls(**response_json)
        except Exception as e:
            logger.error(f"Error in synthesize: {str(e)}")
            raise

    async def load_streaming_response(
        self, response
    ) -> Optional[protocol.ChatStreamingProtocol]:
        """Load and process streaming response"""
        try:
            start_time = time.time()
            streaming_chunks = []
            synapse = None

            async for chunk in response:
                if isinstance(chunk, protocol.ChatStreamingProtocol):
                    synapse = chunk
                    continue
                else:
                    streaming_chunks.append(chunk)
            if len(streaming_chunks) < 0:
                logger.error(
                    f"No streaming chunks found for {synapse.miner_payload.model}"
                )
                return None
            synapse.streaming_chunks = streaming_chunks
            end_time = time.time()
            process_time = end_time - start_time
            synapse.dendrite.process_time = process_time

            chunks_per_second = len(streaming_chunks) / process_time
            logger.success(
                f"Loaded Streaming Response - {len(streaming_chunks)} chunks - {process_time}s "
                f"- {chunks_per_second} chunks/s"
            )
            return synapse
        except Exception as e:
            logger.error(f"Error in load_streaming_response: {str(e)}")
            return None

    async def get_scored_counter(self) -> Dict[int, int]:
        """Get counter of scored UIDs"""
        scored_counter = {}
        async for key in self.redis.scan_iter("scored_uid:*"):
            uid = int(key.decode().split(":")[1])
            count = await self.redis.get(key)
            scored_counter[uid] = int(count)
        return scored_counter

    async def score(
        self,
        uids: List[int],
        responses: List[protocol.ChatStreamingProtocol],
        base_request: protocol.ChatStreamingProtocol,
        batch_id: str,
    ) -> None:
        """Score miner responses"""
        valid_pairs, invalid_pairs = self.response_processor.validate_responses(
            uids, responses
        )

        # Handle invalid responses
        if invalid_pairs:
            invalid_uids = [uid for uid, _, _ in invalid_pairs]
            await self._zero_invalid_miners(invalid_uids)

            # Track invalid responses
            for uid, response, invalid_reason in invalid_pairs:
                tracking_data = ResponseTrackingData(
                    batch_id=batch_id,
                    uid=uid,
                    model=base_request.miner_payload.model,
                    score=0.0,
                    response_time=response.dendrite.process_time if response else 0,
                    invalid_reason=invalid_reason,
                )
                await self._store_tracking_data(tracking_data)

        # Process valid responses
        valid_uids = [uid for uid, _ in valid_pairs]
        valid_responses = [response for _, response in valid_pairs]

        uid_counts = await self.get_scored_counter()
        valid_uids_to_score = [uid for uid in valid_uids if uid_counts.get(uid, 0) < 4]
        logger.info(
            f"valid_uids: {valid_uids} - valid_uids_to_score: {valid_uids_to_score}"
        )
        if not valid_uids_to_score:
            return

        await self._process_valid_responses(
            valid_uids_to_score, valid_responses, base_request, batch_id
        )

    async def _zero_invalid_miners(self, invalid_uids: List[int]) -> None:
        """Set scores to zero for invalid miners"""
        try:
            result = await self.miner_manager_client.post(
                "/api/step",
                json={
                    "scores": [0.0] * len(invalid_uids),
                    "total_uids": invalid_uids,
                },
                timeout=60.0,
            )
            result.raise_for_status()
        except Exception as e:
            logger.error(f"Error zeroing invalid miners: {str(e)}")

    async def _process_valid_responses(
        self,
        valid_uids: List[int],
        valid_responses: List[protocol.ChatStreamingProtocol],
        base_request: protocol.ChatStreamingProtocol,
        batch_id: str,
    ) -> None:
        """Process and score valid responses"""
        try:
            scores = await self._get_response_scores(valid_responses, base_request)
            penalized_scores = self._apply_time_penalties(valid_responses, scores)

            # Track valid responses
            for uid, response, score in zip(
                valid_uids, valid_responses, penalized_scores
            ):
                tracking_data = ResponseTrackingData(
                    batch_id=batch_id,
                    uid=uid,
                    model=base_request.miner_payload.model,
                    score=score,
                    response_time=response.dendrite.process_time,
                )
                await self._store_tracking_data(tracking_data)

            logger.info(
                f"model: {base_request.miner_payload.model} - uids: {valid_uids} - scores: {scores} - penalized_scores: {penalized_scores}"
            )
            if base_request.miner_payload.model == "dall-e-3":
                logger.debug(
                    f"valid_responses: {[r.miner_response for r in valid_responses]}"
                )

            await self._update_miner_scores(valid_uids, penalized_scores)
            await self._update_scoring_records(valid_uids)

        except Exception as e:
            logger.error(f"Error in scoring valid responses: {str(e)}")

    async def _get_response_scores(
        self,
        responses: List[protocol.ChatStreamingProtocol],
        base_request: protocol.ChatStreamingProtocol,
    ) -> List[float]:
        """Get scores for responses from scoring service"""
        score_response = await self.score_client.post(
            "/score",
            json={
                "responses": [r.miner_response for r in responses],
                "request": base_request.miner_payload.model_dump(),
            },
            timeout=60.0,
        )
        return score_response.json()["scores"]

    def _apply_time_penalties(
        self,
        responses: List[protocol.ChatStreamingProtocol],
        scores: List[float],
    ) -> List[float]:
        """Apply time-based penalties to scores"""
        time_penalties = [r.dendrite.process_time / r.timeout for r in responses]
        penalized_scores = [
            max(score - penalty * 0.2, 0)
            for score, penalty in zip(scores, time_penalties)
        ]
        logger.info(f"Original scores: {scores}")
        logger.info(f"Penalized scores: {penalized_scores}")
        return penalized_scores

    async def _update_miner_scores(
        self,
        uids: List[int],
        scores: List[float],
    ) -> None:
        """Update miner scores in the manager"""
        await self.miner_manager_client.post(
            "/api/step",
            json={
                "scores": scores,
                "total_uids": uids,
            },
        )

    async def _update_scoring_records(self, uids: List[int]) -> None:
        """Update scoring records in Redis"""
        pipe = self.redis.pipeline()
        for uid in uids:
            key = f"scored_uid:{uid}"
            pipe.incr(key)
            pipe.expire(key, 360)
        await pipe.execute()

    async def _store_tracking_data(self, data: ResponseTrackingData) -> None:
        """Store response tracking data in Redis"""
        key = f"tracking:{data.batch_id}:{data.uid}"
        value = {
            "batch_id": data.batch_id,
            "uid": data.uid,
            "model": data.model,
            "score": data.score,
            "response_time": data.response_time,
            "invalid_reason": data.invalid_reason,
            "timestamp": data.timestamp,
        }
        await self.redis.hmset(key, value)
        await self.redis.expire(key, 86400)  # Expire after 24 hours


if __name__ == "__main__":
    validator = Validator()
    asyncio.run(validator.run())
