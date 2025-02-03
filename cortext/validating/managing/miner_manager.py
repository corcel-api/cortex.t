import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from loguru import logger
import numpy as np
import bittensor as bt
from .sql_schemas import Base, MinerMetadata
from .serving_counter import ServingCounter
from ...utilities.secure_request import get_headers
from ...global_config import CONFIG
from ...protocol import Credit
import asyncio
import httpx
import traceback


class MinerManager:
    def __init__(self, network: str, netuid: int, wallet_name: str, wallet_hotkey: str):
        self.subtensor = bt.subtensor(network=network)
        self.metagraph = self.subtensor.metagraph(netuid=netuid)
        self.wallet = bt.wallet(name=wallet_name, hotkey=wallet_hotkey)
        self.subtensor_client = httpx.AsyncClient(
            base_url=f"http://{CONFIG.w_subtensor.host}:{CONFIG.w_subtensor.port}",
        )
        self.uid = 0
        self.dendrite = bt.Dendrite(wallet=self.wallet)
        logger.info(f"Connecting to Redis at {CONFIG.redis.host}:{CONFIG.redis.port}")
        self.redis_client = redis.Redis(
            host=CONFIG.redis.host, port=CONFIG.redis.port, db=CONFIG.redis.db
        )
        logger.info(f"Creating SQL engine with URL: {CONFIG.sql.url}")
        self.engine = create_engine(CONFIG.sql.url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        logger.info("Initializing serving counters")
        asyncio.run(self._sync_serving_counter_loop())
        logger.info("Creating background task for serving counter sync")
        asyncio.get_event_loop().create_task(
            self.run_task_in_background(self._sync_serving_counter_loop, 600)
        )
        logger.success("MinerManager initialization complete")

    async def sync_credit(self):
        uids_request = await self.subtensor_client.post("/api/uids", timeout=4, json={})
        uids = uids_request.json()["uids"]
        axons_request = await self.subtensor_client.post(
            "/api/axons", timeout=4, json={"uids": uids}
        )
        axons = axons_request.json()["axons"]
        axons = [bt.AxonInfo.from_string(axon) for axon in axons]
        responses = await self.dendrite.forward(
            axons=axons, synapse=Credit(), timeout=4
        )
        metadata = self.query(uids)
        credits = []
        for uid, response in zip(uids, responses):
            metadata[uid].set_credit(response.credit)
            credits.append(metadata[uid].credit)
        self.credits = credits
        self.uids = uids
        self.session.commit()

    async def run_task_in_background(self, task, repeat_interval: int = 600):
        while True:
            await asyncio.sleep(repeat_interval)
            await task()

    async def _sync_serving_counter_loop(self):
        try:
            logger.info("Syncing serving counter loop")
            uids_request = await self.subtensor_client.post(
                "/api/uids", timeout=4, json={}
            )
            uids = uids_request.json()["uids"]
            await self.sync_credit()
            metadata = self.query(uids)
            percentage_rate_limit_request = await self.subtensor_client.post(
                "/api/rate_limit_percentage",
                timeout=4,
                json={"uid": self.uid},
            )
            percentage_rate_limit = percentage_rate_limit_request.json()[
                "rate_limit_percentage"
            ]
            logger.info(f"Percentage rate limit: {percentage_rate_limit}")
            logger.info(f"Creating serving counters for {len(uids)} UIDs")
            self.serving_counters = {
                uid: ServingCounter(
                    quota=int(metadata[uid].credit * percentage_rate_limit),
                    uid=uid,
                    redis_client=self.redis_client,
                )
                for uid in uids
            }
            logger.success(
                f"Serving counters initialized with rate limit: {self.serving_counters}"
            )
            await self.post_metadata()
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in sync serving counter loop: {e}")
            await asyncio.sleep(600)

    def query(self, uids: list[int] = []) -> dict[int, MinerMetadata]:
        logger.debug(f"Querying metadata for UIDs: {uids if uids else 'all'}")
        query = self.session.query(MinerMetadata)
        if uids:
            query = query.filter(MinerMetadata.uid.in_(uids))
        result = {miner.uid: miner for miner in query.all()}

        if uids:
            for uid in uids:
                if uid not in result:
                    logger.info(f"Creating default metadata for UID {uid}")
                    miner = MinerMetadata(uid=uid)
                    self.session.add(miner)
                    result[uid] = miner
            self.session.commit()

        logger.success(f"Found {len(result)} miner metadata records")
        return result

    def consume(self, threshold: float, k: int, task_credit: int):
        logger.info(
            f"Starting credit consumption process: {task_credit} credit for {k} miners"
        )

        # Get remaining quotas atomically using pipeline
        logger.debug("Initializing Redis pipeline to fetch remaining quotas.")
        pipe = self.redis_client.pipeline()
        for uid in self.uids:
            logger.debug(f"Fetching quota for UID {uid}.")
            pipe.get(self.serving_counters[uid].key)
            pipe.get(self.serving_counters[uid].quota_key)
        results = pipe.execute()
        logger.debug(f"Pipeline execution results: {results}")

        # Calculate remaining quotas and probabilities
        remaining_quotas = []
        for i in range(0, len(results), 2):
            current_count = int(results[i] or 0)
            quota = int(results[i + 1] or 0)
            remaining = max(0, quota - current_count)
            remaining_quotas.append(remaining)
            logger.debug(
                f"UID {self.uids[i // 2]}: Current count = {current_count}, Quota = {quota}, Remaining = {remaining}"
            )

        total_remaining = sum(remaining_quotas)
        logger.info(f"Remaining quotas: {remaining_quotas}")
        logger.info(f"Total remaining quota across all UIDs: {total_remaining}")
        if total_remaining == 0:
            logger.warning("No remaining quota available for any UID.")
            return []

        probabilities = np.array(remaining_quotas) / total_remaining
        max_available_uid = len([p for p in probabilities if p > 0])
        k = min(k, max_available_uid)
        logger.info(
            f"Calculated probabilities for UIDs. Max available UIDs: {max_available_uid}, Adjusted k: {k}"
        )

        # Select UIDs based on remaining quota probabilities
        uids = np.random.choice(
            self.uids, size=k, replace=False, p=probabilities
        ).tolist()
        logger.info(f"Selected UIDs for consumption: {uids}")

        # Attempt to consume atomically
        consume_results = []
        for uid in uids:
            logger.debug(f"Attempting to consume credit for UID {uid}.")
            consume_results.append(
                self.serving_counters[uid].increment(task_credit, threshold)
            )

        # Filter successful consumptions
        uids = [uid for uid, result in zip(uids, consume_results) if result]
        logger.info(f"Successfully consumed {task_credit} credit for UIDs: {uids}.")
        return uids

    def step(self, scores: list[float], total_uids: list[int]):
        logger.info(f"Updating scores for {len(total_uids)} miners")
        credits = [self.credits[uid] for uid in total_uids]
        credits = np.array(credits)
        credit_scales = np.array(credits) / CONFIG.bandwidth.max_credit
        credit_scales[credit_scales > 1] = 1
        logger.info(f"Credit scales: {credit_scales}")
        miners = self.query(total_uids)
        for uid, score, credit_scale in zip(total_uids, scores, credit_scales):
            logger.info(
                f"Processing UID {uid} with score {score}:score*credit_scale:{credit_scale}"
            )
            score = score * credit_scale
            miner = miners[uid]
            # EMA with decay factor
            miner.accumulate_score = (
                miner.accumulate_score * CONFIG.score.decay_factor
                + score * (1 - CONFIG.score.decay_factor)
            )
            miner.accumulate_score = max(0, miner.accumulate_score)
            logger.info(
                f"Updated accumulate_score for UID {uid}: {miner.accumulate_score}"
            )
        self.session.commit()
        logger.success(f"Updated metadata for {len(total_uids)} uids")

    @property
    def weights(self):
        try:
            uids = []
            scores = []
            for uid, miner in self.query().items():
                uids.append(uid)
                scores.append(miner.accumulate_score)

            scores = np.array(scores)
            if scores.sum() > 0:
                scores = scores / scores.sum()
            else:
                scores = np.zeros_like(scores)
            return uids, scores.tolist()
        except Exception as e:
            logger.error(f"Error in weights: {e}")
            return [], []

    async def post_metadata(self):
        try:
            logger.info("Posting metadata")
            headers = get_headers(self.dendrite.keypair)
            logger.debug(f"Headers: {headers}")
            metadata = self.query()
            metadata = {uid: miner.to_dict() for uid, miner in metadata.items()}
            logger.info(f"Metadata: {metadata}")
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{CONFIG.subnet_report_url}/api/report_metadata",
                    timeout=4,
                    json=metadata,
                    headers=headers,
                )
            logger.debug(f"Response: {response}")
        except Exception as e:
            logger.error(f"Error in post metadata: {e}")
            return
