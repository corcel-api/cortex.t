from pydantic import BaseModel, Field, validator
from bittensor import StreamingSynapse, Synapse
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from starlette.responses import StreamingResponse
from typing import Optional
from .global_config import CONFIG


class Credit(Synapse):
    credit: int = Field(default=0, description="Credit of miner")

    @validator("credit")
    def validate_credit(cls, v):
        return max(min(v, CONFIG.bandwidth.max_credit), CONFIG.bandwidth.min_credit)


class MinerPayload(BaseModel):
    model: str = Field(
        description="The model to be used for the miner", default="gpt-4o-mini"
    )
    messages: list[dict] = Field(
        description="The messages to be sent to the miner", default=[]
    )
    temperature: float = Field(
        description="The temperature to be used for the miner", default=0.0
    )
    max_tokens: int = Field(
        description="The maximum number of tokens to be used for the miner",
        default=4096,
    )
    top_p: float = Field(description="The top p to be used for the miner", default=1.0)
    frequency_penalty: float = Field(
        description="The frequency penalty to be used for the miner", default=0.0
    )
    stream: bool = Field(description="Whether to stream the response", default=True)


class ChoiceDelta(BaseModel):
    content: Optional[str] = None
    """The contents of the chunk message."""


class Choice(BaseModel):
    index: int = Field(description="The index of the choice")
    delta: ChoiceDelta = Field(description="The delta of the choice")
    logprobs: Optional[dict] = Field(description="The logprobs of the choice")
    finish_reason: Optional[str] = Field(description="The finish reason of the choice")


class MinerResponse(BaseModel):
    id: str = Field(description="The ID of the response")
    choices: list[Choice] = Field(description="The choices from the response")
    created: int = Field(description="When the response was created")
    model: str = Field(description="The model used for the response")
    object: str = Field(description="The object type")


class ScoringRequest(BaseModel):
    responses: list[str]
    request: MinerPayload


class ScoringResponse(BaseModel):
    scores: list[float] = Field(description="The scores of the responses", default=[])


class ChatStreamingProtocol(StreamingSynapse):
    miner_payload: MinerPayload = Field(
        description="The payload for the miner. Can not modify this field",
        default=MinerPayload(),
        frozen=True,
    )
    streaming_chunks: list[MinerResponse] = Field(
        description="The response from the miner", default=[]
    )

    @property
    def miner_response(self):
        return "".join([r.choices[0].delta.content for r in self.streaming_chunks])

    @property
    def completion(self):
        return "".join([r.choices[0].delta.content for r in self.streaming_chunks])

    async def process_streaming_response(self, response: StreamingResponse):
        async for line in response.content:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:].strip()  # Remove 'data: ' prefix
                if data == "[DONE]":
                    print("DONE")
                    break
                try:
                    chunk = MinerResponse.model_validate_json(data)
                    self.streaming_chunks.append(chunk)
                    yield chunk
                except Exception as e:
                    print("Error", e)
                    break  # Skip invalid chunks

    def extract_response_json(self, response: StreamingSynapse) -> dict:
        # iterate over the response headers and extract the necessary data
        headers = {
            k.decode("utf-8"): v.decode("utf-8")
            for k, v in response.__dict__["_raw_headers"]
        }

        # helper function to extract data from headers
        def extract_info(prefix):
            return {
                key.split("_")[-1]: value
                for key, value in headers.items()
                if key.startswith(prefix)
            }

        # return the extracted data in the expected format
        return {
            "name": headers.get("name", ""),
            "timeout": float(headers.get("timeout", 0)),
            "total_size": int(headers.get("total_size", 0)),
            "header_size": int(headers.get("header_size", 0)),
            "dendrite": extract_info("bt_header_dendrite"),  # dendrite info
            "axon": extract_info("bt_header_axon"),  # axon info
            "miner_responses": self.miner_responses,
            "miner_payload": self.miner_payload,
        }

    def verify(self) -> bool:
        if len(self.miner_responses) == 0:
            return False
        completion = ""
        for response in self.miner_responses:
            completion += response.choices[0].delta.content
        if not len(completion):
            return False
        return True
