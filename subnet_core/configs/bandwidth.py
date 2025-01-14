from pydantic import BaseModel, Field
import random


class ModelConfig(BaseModel):
    credit: int
    model: str
    max_tokens: int
    synapse_type: str
    timeout: int


class BandwidthConfig(BaseModel):
    interval: int = Field(default=60)
    min_stake: int = Field(default=10000)
    model_cost: dict[str, ModelConfig] = {
        "gpt-4o": ModelConfig(
            credit=4,
            model="gpt-4o",
            max_tokens=8096,
            synapse_type="streaming-chat",
            timeout=12,
        ),
        "claude-3-5-sonnet": ModelConfig(
            credit=2,
            model="claude-3-5-sonnet",
            max_tokens=8096,
            synapse_type="streaming-chat",
            timeout=12,
        ),
    }
    min_credit: int = Field(default=128)
    max_credit: int = Field(default=1024)

    @property
    def sample_model(self) -> ModelConfig:
        models = list(self.model_cost.values())
        weights = [model.credit for model in models]
        return random.choices(models, weights=weights, k=1)[0]
