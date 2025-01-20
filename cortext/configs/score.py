from pydantic import BaseModel, Field


class ScoreConfig(BaseModel):
    decay_factor: float = Field(default=0.9)
    host: str = Field(default="localhost")
    port: int = Field(default=8200)
