import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from loguru import logger
from sqlalchemy.ext.declarative import declarative_base
import numpy as np
import bittensor as bt
import random
from .sql_schemas import Base, MinerMetadata
from .serving_counter import ServingCounter
from ...global_config import CONFIG
from ...ultilities.rate_limit import get_rate_limit_proportion
from ...protocol import Credit
import asyncio


class MinerManager:
    def __init__(self, uid, metagraph, wallet):
        logger.info(f"Initializing MinerManager with uid: {uid}")
        self.uid = uid
        self.metagraph = metagraph
        self.wallet = wallet
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
        uids = self.metagraph.uids.tolist()
        axons = self.metagraph.axons
        responses = await self.dendrite.forward(
            axons=axons,
            synapse=Credit(),
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
            uids = self.metagraph.uids.tolist()
            await self.sync_credit()
            metadata = self.query(uids)
            percentage_rate_limit = get_rate_limit_proportion(self.metagraph, self.uid)
            logger.info(f"Percentage rate limit: {percentage_rate_limit}")
            if CONFIG.network == "testnet":
                percentage_rate_limit = 1
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
        except Exception as e:
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
        logger.info(f"Consuming {task_credit} credit for {k} miners")
        logger.info(f"Credits: {self.credits}")
        uids = random.choices(self.uids, weights=self.credits, k=k)
        consume_results = []
        for uid in uids:
            consume_results.append(
                self.serving_counters[uid].increment(task_credit, threshold)
            )
        uids = [uid for uid, result in zip(uids, consume_results) if result]
        logger.info(f"Consumed {task_credit} credit for {uids}.")
        return uids

    def step(self, scores: list[float], total_uids: list[int]):
        logger.info(f"Updating scores for {len(total_uids)} miners")
        miners = self.query(total_uids)
        for uid, score in zip(total_uids, scores):
            logger.info(f"Processing UID {uid} with score {score}")
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
        uids = []
        scores = []
        for uid, miner in self.query().items():
            uids.append(uid)
            scores.append(miner.accumulate_score)

        scores = np.array(scores)
        scores = scores / scores.sum()
        credit_scales = np.array(self.credits) / CONFIG.bandwidth.max_credit
        credit_scales[credit_scales > 1] = 1
        scores = scores * credit_scales
        return uids, scores
