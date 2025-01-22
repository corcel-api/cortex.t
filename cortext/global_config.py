from pydantic_settings import BaseSettings
from .configs import (
    RedisConfig,
    BandwidthConfig,
    ScoreConfig,
    SQLConfig,
    SynthesizeConfig,
    MinerManagerConfig,
    ValidatingConfig,
    WSubtensorConfig,
    OrganicConfig,
)
from loguru import logger
from dotenv import load_dotenv
import json

load_dotenv()


class GlobalConfig(BaseSettings):
    redis: RedisConfig = RedisConfig(host="localhost", port=6379)
    bandwidth: BandwidthConfig = BandwidthConfig(host="localhost", port=8100)
    score: ScoreConfig = ScoreConfig(host="localhost", port=8101)
    sql: SQLConfig = SQLConfig(url="sqlite:///miner_metadata.db")
    network: str = "mainnet"
    synthesize: SynthesizeConfig = SynthesizeConfig(host="localhost", port=8102)
    miner_manager: MinerManagerConfig = MinerManagerConfig(host="localhost", port=8103)
    validating: ValidatingConfig = ValidatingConfig(
        synthetic_threshold=0.1,
        synthetic_batch_size=4,
        synthetic_concurrent_batches=4,
    )
    w_subtensor: WSubtensorConfig = WSubtensorConfig(host="localhost", port=8104)
    organic: OrganicConfig = OrganicConfig(host="localhost", port=8105)
    subtensor_network: str = "finney"
    subtensor_netuid: int = 18
    wallet_name: str = "default"
    wallet_hotkey: str = "default"
    subtensor_tempo: int = 360
    axon_port: int = 8000

    class Config:
        env_nested_delimiter = "__"


CONFIG = GlobalConfig()
logger.info("\n" + json.dumps(CONFIG.model_dump(), indent=2))
