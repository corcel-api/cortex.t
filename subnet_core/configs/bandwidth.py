from pydantic import BaseModel, Field


class ModelCost(BaseModel):
    credit: int
    model_name: str


class BandwidthConfig(BaseModel):
    interval: int = Field(default=60)
    min_stake: int = Field(default=10000)
    model_cost: dict[str, ModelCost] = {
        "gpt-4o": ModelCost(credit=4, model_name="gpt-4o"),
        "gpt-4o-mini": ModelCost(credit=1, model_name="gpt-4o-mini"),
    }
    min_credit: int = Field(default=128)
    max_credit: int = Field(default=1024)
