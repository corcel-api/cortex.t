from pydantic import BaseModel


class OrganicConfig(BaseModel):
    host: str
    port: int
