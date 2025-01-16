from pydantic import BaseModel


class SynthesizeConfig(BaseModel):
    host: str
    port: int
    synthetic_pool_size: int = 1024
    organic_pool_size: int = 1024
    