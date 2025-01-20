from pydantic import BaseModel, Field


class SQLConfig(BaseModel):
    url: str = Field(default="sqlite:///miner_metadata.db")
