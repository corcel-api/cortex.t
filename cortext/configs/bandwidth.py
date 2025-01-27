from pydantic import BaseModel, Field
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
        models = list(self.model_configs.values())
        weights = [model.credit for model in models]
        return random.choices(models, weights=weights, k=1)[0]
