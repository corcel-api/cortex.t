from pydantic import BaseModel, Field
import os


class RedisConfig(BaseModel):
    host: str
    port: int
    db: int
    organic_queue_key: str
    synthetic_queue_key: str
    miner_manager_key: str
