from pydantic import BaseModel
from typing import List


class UIDsRequest(BaseModel):
    uids: List[int]


class UIDsResponse(BaseModel):
    uids: List[int]


class AxonsRequest(BaseModel):
    uids: List[int]


class AxonsResponse(BaseModel):
    axons: List[str]


class RateLimitRequest(BaseModel):
    uid: int


class RateLimitResponse(BaseModel):
    rate_limit_percentage: float


class SetWeightsResponse(BaseModel):
    success: bool
    message: str
