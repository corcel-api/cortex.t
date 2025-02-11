from pydantic import BaseModel
import random


class ModelConfig(BaseModel):
    credit: int
    model: str
    max_tokens: int
    synapse_type: str
    timeout: int
    allowed_params: list[str]


class BandwidthConfig(BaseModel):
    interval: int
    min_stake: int
    model_configs: dict[str, ModelConfig]
    min_credit: int
    max_credit: int

    @property
    def sample_model(self) -> ModelConfig:
        return random.choice(list(self.model_configs.values()))
