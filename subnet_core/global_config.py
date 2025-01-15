from pydantic_settings import BaseSettings
from .configs import (
    RedisConfig,
    BandwidthConfig,
    ScoreConfig,
    SQLConfig,
    SynthesizeConfig,
    MinerManagerConfig,
)
from loguru import logger


class GlobalConfig(BaseSettings):
    redis: RedisConfig = RedisConfig()
    bandwidth: BandwidthConfig = BandwidthConfig()
    score: ScoreConfig = ScoreConfig()
    sql: SQLConfig = SQLConfig(url="sqlite:///miner_metadata.db")
    network: str = "mainnet"
    synthesize: SynthesizeConfig = SynthesizeConfig(host="localhost", port=8887)
    miner_manager: MinerManagerConfig = MinerManagerConfig()

    class Config:
        env_nested_delimiter = "__"


CONFIG = GlobalConfig()
logger.info(f"GlobalConfig: {CONFIG}")
