from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.responses import StreamingResponse
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk


# Definitions
class MinerPayload(BaseModel):
    model: str = ""
    messages: List[dict] = []
    temperature: float = 0.0
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    stream: bool = True


class MinerResponse(ChatCompletionChunk):
    pass


class ScoringResponse(BaseModel):
    score: float = 0.0


# FastAPI Server
app = FastAPI()


@app.post("/chat-stream")
async def chat_stream(payload: MinerPayload):
    """
    Handles chat streaming requests.
    """
    if not payload.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")

    def stream_response():
        """Simulates a streaming response."""
        for i in range(5):
            yield f"data: Simulated chunk {i} for model {payload.model}\n"

    if payload.stream:
        return StreamingResponse(stream_response(), media_type="text/event-stream")
    else:
        return MinerResponse(choices=[{"delta": {"content": "Simulated response"}}])


@app.post("/score")
async def score_response():
    """
    Handles scoring requests.
    """
    return ScoringResponse(score=0.85)
