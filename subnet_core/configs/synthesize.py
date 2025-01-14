from pydantic import BaseModel


class SynthesizeConfig(BaseModel):
    host: str
    port: int
