from pydantic import BaseModel


class WSubtensorConfig(BaseModel):
    host: str
    port: int
