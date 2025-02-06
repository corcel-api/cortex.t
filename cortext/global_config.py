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
    ModelConfig,
)
from rich import print as rprint
from dotenv import load_dotenv

load_dotenv()


class GlobalConfig(BaseSettings):
    redis: RedisConfig = RedisConfig(
        host="localhost",
        port=6379,
        db=0,
        organic_queue_key="organic_queue",
        synthetic_queue_key="synthetic_queue",
        miner_manager_key="node_manager",
    )
    bandwidth: BandwidthConfig = BandwidthConfig(
        interval=60,
        min_stake=10000,
        model_configs={
            "gpt-4o": ModelConfig(
                credit=4,
                model="gpt-4o",
                max_tokens=8096,
                synapse_type="streaming-chat",
                timeout=32,
                allowed_params=[
                    "messages",
                    "temperature",
                    "max_tokens",
                    "stream",
                    "model",
                    "seed",
                ],
            ),
            "gpt-4o-mini": ModelConfig(
                credit=1,
                model="gpt-4o-mini",
                max_tokens=8096,
                synapse_type="streaming-chat",
                timeout=32,
                allowed_params=[
                    "messages",
                    "temperature",
                    "max_tokens",
                    "stream",
                    "model",
                    "seed",
                ],
            ),
            "dall-e-3": ModelConfig(
                credit=2,
                model="dall-e-3",
                timeout=32,
                synapse_type="streaming-chat",
                max_tokens=1024,
                allowed_params=["prompt", "n", "size", "response_format", "user"],
            ),
            "claude-3-5-sonnet-20241022": ModelConfig(
                credit=4,
                model="claude-3-5-sonnet-20241022",
                timeout=32,
                synapse_type="streaming-chat",
                max_tokens=8096,
                allowed_params=[
                    "messages",
                    "temperature",
                    "max_tokens",
                    "stream",
                    "model",
                ],
            ),
        },
        min_credit=48,
        max_credit=256,
    )
    score: ScoreConfig = ScoreConfig(host="localhost", port=8101, decay_factor=0.9)
    sql: SQLConfig = SQLConfig(url="sqlite:///miner_metadata.db")
    network: str = "mainnet"
    synthesize: SynthesizeConfig = SynthesizeConfig(
        host="localhost",
        port=8102,
        synthetic_pool_size=8096,
        organic_pool_size=1024,
    )
    miner_manager: MinerManagerConfig = MinerManagerConfig(host="localhost", port=8103)
    validating: ValidatingConfig = ValidatingConfig(
        synthetic_threshold=0.2,
        synthetic_batch_size=4,
        synthetic_concurrent_batches=1,
    )
    w_subtensor: WSubtensorConfig = WSubtensorConfig(host="localhost", port=8104)
    organic: OrganicConfig = OrganicConfig(host="localhost", port=8105)
    subtensor_network: str = "finney"
    subtensor_netuid: int = 18
    wallet_name: str = "default"
    wallet_hotkey: str = "default"
    subtensor_tempo: int = 360
    axon_port: int = 8000
    subnet_report_url: str = "https://cortext-subnet-report.corcel.io"
    weight_version: int = 2**64 - 7

    class Config:
        env_nested_delimiter = "__"


CONFIG = GlobalConfig()

rprint(CONFIG.model_dump())
