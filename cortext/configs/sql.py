from pydantic import BaseModel, Field


class SQLConfig(BaseModel):
    url: str
