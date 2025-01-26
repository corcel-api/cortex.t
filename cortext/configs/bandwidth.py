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
    interval: int = Field(default=60)
    min_stake: int = Field(default=10000)
    model_configs: dict[str, ModelConfig] = {
        "gpt-4o": ModelConfig(
            credit=1,
            model="gpt-4o",
            max_tokens=8096,
            synapse_type="streaming-chat",
            timeout=12,
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
            credit=1,
            model="dall-e-3",
            timeout=12,
            synapse_type="streaming-chat",
            max_tokens=1024,
            allowed_params=["prompt", "n", "size", "response_format", "user"],
        ),
        "claude-3-5-sonnet-20241022": ModelConfig(
            credit=1,
            model="claude-3-5-sonnet-20241022",
            timeout=12,
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
    }
    min_credit: int = Field(default=128)
    max_credit: int = Field(default=1024)

    @property
    def sample_model(self) -> ModelConfig:
        models = list(self.model_configs.values())
        weights = [model.credit for model in models]
        return random.choices(models, weights=weights, k=1)[0]
