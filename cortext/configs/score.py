from pydantic import BaseModel, Field


class ScoreConfig(BaseModel):
    decay_factor: float
    host: str
    port: int
