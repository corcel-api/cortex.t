from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI
from cortext import CONFIG, protocol
from fastapi.responses import StreamingResponse
import httpx
import bittensor as bt
from loguru import logger
from redis.asyncio import Redis
import uvicorn

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


@app.post("/api/v1/chat/completions")
async def chat_completions(request: protocol.MinerPayload):
    logger.info(f"Received request: {request}")
    try:
        # Get miner UID
        response = await managing_client.post(
            "/api/consume",
            json={
                "threshold": 0.9,
                "k": 1,
                "task_credit": 1,
            },
        )
        uid = response.json()["uids"][0]
        uid = 1
        logger.info(f"Consumed {uid} miner")
        # Push request to redis queue using await
        await redis_client.rpush(
            CONFIG.redis.organic_queue_key, request.model_dump_json()
        )
        # Create synapse and forward request
        synapse = protocol.ChatStreamingProtocol(
            miner_payload=request,
        )
        axon_data = await subtensor_client.post("/api/axons", json=[uid])
        axon = bt.AxonInfo.from_string(axon_data.json()[0])
        axon.ip = "149.28.139.101"
        axon.port = 9999
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
            except Exception as e:
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
        logger.error(f"Request error: {e}")
        return {"error": {"message": str(e), "type": "request_error"}}


if __name__ == "__main__":
    uvicorn.run(app, host=CONFIG.organic.host, port=CONFIG.organic.port)
