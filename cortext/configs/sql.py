from pydantic import BaseModel


class SQLConfig(BaseModel):
    url: str
