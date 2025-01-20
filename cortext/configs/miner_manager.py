from pydantic import BaseModel


class MinerManagerConfig(BaseModel):
    port: int = 8500
    host: str = "localhost"
