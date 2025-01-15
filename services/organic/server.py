from fastapi import FastAPI
from dotenv import load_dotenv
from subnet_core import CONFIG, protocol
from fastapi.responses import StreamingResponse
import httpx
import bittensor as bt
import os
from loguru import logger

load_dotenv()

managing_client = httpx.AsyncClient(
    base_url=f"http://{CONFIG.miner_manager.host}:{CONFIG.miner_manager.port}"
)

subtensor = bt.Subtensor(network=os.getenv("SUBTENSOR.NETWORK"))
metagraph = subtensor.metagraph(os.getenv("NETUID"))
wallet = bt.wallet(
    name=os.getenv("WALLET.NAME"),
    hotkey=os.getenv("WALLET.HOTKEY"),
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
        logger.info(f"Consumed {uid} miner")
        uid = 1

        # Create synapse and forward request
        synapse = protocol.ChatStreamingProtocol(
            miner_payload=request,
        )
        axon = metagraph.axons[uid]
        responses = await dendrite.forward(
            axons=[axon], synapse=synapse, streaming=True, timeout=12
        )
        response = responses[0]

        async def stream_response():
            try:
                async for chunk in response:
                    if not isinstance(chunk, protocol.MinerResponse):
                        continue
                    print(chunk)
                    # Format response to match OpenAI streaming format
                    yield f"data: {chunk.model_dump_json()}\n\n"
                # Send final [DONE] message
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
