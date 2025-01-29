from ...global_config import CONFIG
import redis
from loguru import logger


class ServingCounter:
    def __init__(
        self,
        quota: int,
        uid: int,
        redis_client: redis.Redis,
        postfix_key: str = "",
    ):
        self.quota = quota
        self.redis_client = redis_client
        self.key = f":{CONFIG.redis.miner_manager_key}:{postfix_key}:{uid}"
        self.quota_key = f"{CONFIG.redis.miner_manager_key}:{postfix_key}:quota:{uid}"
        self.redis_client.set(self.quota_key, quota)

    def increment(self, amount: int = 1, ignore_threshold: float = None) -> bool:
        """
        Increment request counter and check rate limit.

        Uses atomic Redis INCR operation and sets expiry on first increment.

        Reset the counter after EPOCH_LENGTH seconds.

        Returns:
            bool: True if under rate limit, False if exceeded
        """
        if ignore_threshold is not None:
            current_count = self.redis_client.get(self.key)
            if current_count is None:
                current_count = 0
            else:
                current_count = int(current_count)
            consumed_proportion = current_count / self.quota
            if consumed_proportion >= ignore_threshold:
                logger.info(
                    f"Rate limit exceeded for {self.key} with threshold {ignore_threshold}: consumed {consumed_proportion * 100}%"
                )
                return False
        count = self.redis_client.incr(self.key, amount)

        if count == amount:
            logger.info(
                f"Setting expiry for {self.key} to {CONFIG.bandwidth.interval} seconds"
            )
            self.redis_client.expire(self.key, CONFIG.bandwidth.interval)

        if count <= self.quota:
            logger.info(f"Consumed {count} of {self.quota} for {self.key}")
            return True

        logger.info(
            f"Rate limit exceeded for {self.key}: consumed {count / self.quota * 100}%"
        )
        return False

    def __repr__(self):
        return f"ServingCounter(quota={self.quota}, key={self.key})"
