from pydantic import BaseModel


class MinerManagerConfig(BaseModel):
    port: int
    host: str
