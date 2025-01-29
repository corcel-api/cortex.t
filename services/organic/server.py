from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from decimal import Decimal
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from cortext import CONFIG, protocol
from fastapi.responses import StreamingResponse
import httpx
import bittensor as bt
from loguru import logger
from redis.asyncio import Redis
import uvicorn
import secrets
import os
from dateutil.relativedelta import relativedelta
import traceback

# Replace api_key_header with security scheme
security = HTTPBearer()

# Get admin API key from environment variable with fallback to test key
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "test-key")
API_KEYS = {ADMIN_API_KEY: "admin"}  # Initialize with admin key

managing_client = httpx.AsyncClient(
    base_url=f"http://{CONFIG.miner_manager.host}:{CONFIG.miner_manager.port}"
)
redis_client = Redis(host=CONFIG.redis.host, port=CONFIG.redis.port, db=CONFIG.redis.db)
wallet = bt.wallet(
    name=CONFIG.wallet_name,
    hotkey=CONFIG.wallet_hotkey,
)
subtensor_client = httpx.AsyncClient(
    base_url=f"http://{CONFIG.w_subtensor.host}:{CONFIG.w_subtensor.port}"
)
dendrite = bt.Dendrite(wallet=wallet)
app = FastAPI()


class APIKey(BaseModel):
    key: str
    user_id: str
    created_at: datetime
    is_active: bool
    permissions: list[str] = ["chat"]
    total_credits: Decimal = Field(default=0.0, ge=0)  # Total credits allocated
    used_credits: Decimal = Field(default=0.0, ge=0)  # Credits used
    credit_reset_date: Optional[datetime] = None  # For recurring credit allowance


async def store_api_key(redis_client: Redis, api_key: APIKey):
    await redis_client.hset(
        f"api_key:{api_key.key}",
        mapping={
            "user_id": api_key.user_id,
            "created_at": api_key.created_at.isoformat(),
            "is_active": str(api_key.is_active),
            "permissions": ",".join(api_key.permissions),
            "total_credits": str(api_key.total_credits),
            "used_credits": str(api_key.used_credits),
            "credit_reset_date": (
                api_key.credit_reset_date.isoformat()
                if api_key.credit_reset_date
                else ""
            ),
        },
    )


async def get_api_key(redis_client: Redis, key: str) -> Optional[APIKey]:
    data = await redis_client.hgetall(f"api_key:{key}")
    if not data:
        return None
    return APIKey(
        key=key,
        user_id=data[b"user_id"].decode(),
        created_at=datetime.fromisoformat(data[b"created_at"].decode()),
        is_active=data[b"is_active"].decode() == "True",
        permissions=data[b"permissions"].decode().split(","),
        total_credits=Decimal(data[b"total_credits"].decode()),
        used_credits=Decimal(data[b"used_credits"].decode()),
        credit_reset_date=(
            datetime.fromisoformat(data[b"credit_reset_date"].decode())
            if data[b"credit_reset_date"]
            else None
        ),
    )


async def update_credit_usage(redis_client: Redis, api_key: str, credits_used: Decimal):
    key_data = await get_api_key(redis_client, api_key)
    if key_data:
        key_data.used_credits += credits_used
        await store_api_key(redis_client, key_data)


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    redis: Redis = Depends(lambda: redis_client),
):
    logger.info(credentials)
    api_key = credentials.credentials  # Extract token from Bearer credentials
    key_data = await get_api_key(redis, api_key)

    logger.info(f"Key data: {key_data}")

    if not key_data or not key_data.is_active:
        raise HTTPException(
            status_code=401,
            detail="Invalid or inactive API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if "chat" not in key_data.permissions:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions",
        )

    # Check if credits need to be reset
    if key_data.credit_reset_date and datetime.utcnow() >= key_data.credit_reset_date:
        key_data.used_credits = Decimal(0)
        key_data.credit_reset_date = calculate_next_reset_date(
            key_data.credit_reset_date
        )
        await store_api_key(redis, key_data)

    # Check if enough credits are available
    remaining_credits = key_data.total_credits - key_data.used_credits
    if remaining_credits <= 0:
        raise HTTPException(
            status_code=403,
            detail="Insufficient credits",
        )

    return key_data


async def chat_completions(
    request: protocol.MinerPayload, api_key: APIKey = Depends(verify_api_key)
):
    logger.info(f"Received request: {request}")
    try:
        # Calculate required credits for this request
        required_credits = CONFIG.bandwidth.model_configs[request.model].credit

        # Check if user has enough credits
        remaining_credits = api_key.total_credits - api_key.used_credits
        if remaining_credits < required_credits:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient credits. Required: {required_credits}, Remaining: {remaining_credits}",
            )

        # Get miner UID
        response = await managing_client.post(
            "/api/consume",
            json={
                "threshold": 0.9,
                "k": 1,
                "task_credit": required_credits,
            },
        )

        # Update credit usage
        await update_credit_usage(redis_client, api_key.key, Decimal(required_credits))

        # uid = response.json()["uids"][0]
        uid = 1
        logger.info(f"Consumed miner uid: {uid}")
        # Create synapse and forward request
        synapse = protocol.ChatStreamingProtocol(
            miner_payload=request,
        )
        axon_data = await subtensor_client.post("/api/axons", json=[uid])
        axon = bt.AxonInfo.from_string(axon_data.json()[0])
        logger.info(f"Forwarding request to {axon}")
        responses = await dendrite.forward(
            axons=[axon], synapse=synapse, streaming=True, timeout=64
        )
        response = responses[0]

        async def stream_response():
            try:
                async for chunk in response:
                    if not isinstance(chunk, protocol.MinerResponse):
                        continue
                    yield f"data: {chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                await redis_client.rpush(
                    CONFIG.redis.organic_queue_key, request.model_dump_json()
                )
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Streaming error: {e}")
                error_response = {
                    "error": {"message": str(e), "type": "streaming_error"}
                }
                yield f"data: {error_response}\n\n"

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Request error: {e}")
        return {"error": {"message": str(e), "type": "request_error"}}


app.add_api_route(
    "/api/v1/chat/completions", chat_completions, methods=["POST", "OPTIONS"]
)


# Utility functions
def calculate_next_reset_date(current_reset_date: datetime) -> datetime:
    """Calculate the next reset date based on your billing cycle logic"""
    return current_reset_date + relativedelta(months=1)


# API Key management endpoints
@app.post("/api/v1/keys")
async def create_key(
    user_id: str,
    initial_credits: float = 100.0,
    monthly_reset: bool = True,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    # Verify admin access using Bearer token
    if credentials.credentials != ADMIN_API_KEY:
        raise HTTPException(
            status_code=403, detail="Admin API key required for this operation"
        )

    reset_date = datetime.utcnow() + relativedelta(months=1) if monthly_reset else None
    api_key = APIKey(
        key=secrets.token_urlsafe(32),
        user_id=user_id,
        created_at=datetime.utcnow(),
        is_active=True,
        permissions=["chat"],
        total_credits=Decimal(initial_credits),
        used_credits=Decimal(0),
        credit_reset_date=reset_date,
    )
    await store_api_key(redis_client, api_key)
    return api_key


@app.post("/api/v1/keys/{key}/add-credits")
async def add_credits(
    key: str,
    amount: float,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if credentials.credentials != ADMIN_API_KEY:
        raise HTTPException(
            status_code=403, detail="Admin API key required for this operation"
        )

    key_data = await get_api_key(redis_client, key)
    if not key_data:
        raise HTTPException(status_code=404, detail="API key not found")

    key_data.total_credits += Decimal(amount)
    await store_api_key(redis_client, key_data)
    return key_data


@app.get("/api/v1/keys", response_model=list[APIKey])
async def get_all_keys(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Verify admin access using Bearer token
    if credentials.credentials != ADMIN_API_KEY:
        raise HTTPException(
            status_code=403, detail="Admin API key required for this operation"
        )

    keys = []
    async for key in redis_client.scan_iter("api_key:*"):
        key_str = key.decode().split(":")[1]
        key_data = await get_api_key(redis_client, key_str)
        if key_data:
            keys.append(key_data)
    return keys


@app.patch("/api/v1/keys/{key}/status")
async def update_key_status(
    key: str,
    status_update: dict,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if credentials.credentials != ADMIN_API_KEY:
        raise HTTPException(
            status_code=403, detail="Admin API key required for this operation"
        )

    key_data = await get_api_key(redis_client, key)
    if not key_data:
        raise HTTPException(status_code=404, detail="API key not found")

    key_data.is_active = status_update.get("is_active", key_data.is_active)
    await store_api_key(redis_client, key_data)
    return key_data


@app.delete("/api/v1/keys/{key}")
async def delete_key(
    key: str,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if credentials.credentials != ADMIN_API_KEY:
        raise HTTPException(
            status_code=403, detail="Admin API key required for this operation"
        )

    deleted = await redis_client.delete(f"api_key:{key}")
    if not deleted:
        raise HTTPException(status_code=404, detail="API key not found")

    return {"status": "success"}


if __name__ == "__main__":
    uvicorn.run(app, host=CONFIG.organic.host, port=CONFIG.organic.port)
