from pydantic import BaseModel


class ScoreConfig(BaseModel):
    decay_factor: float
    host: str
    port: int
