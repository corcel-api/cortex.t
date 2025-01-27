from pydantic import BaseModel, Field, validator
from bittensor import StreamingSynapse, Synapse
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from starlette.responses import StreamingResponse
from typing import Optional
from .global_config import CONFIG
import json
import base64
from loguru import logger


class Credit(Synapse):
    credit: int = Field(default=0, description="Credit of miner")

    @validator("credit")
    def validate_credit(cls, v):
        return max(min(v, CONFIG.bandwidth.max_credit), CONFIG.bandwidth.min_credit)


class MinerPayload(BaseModel):
    model: str = Field(description="The model to be used for the miner", default="")
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
    stream: bool = Field(description="Whether to stream the response", default=True)
    seed: int = Field(description="The seed to be used for the miner", default=42)


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
    system_fingerprint: Optional[str] = Field(
        description="The system fingerprint of the choice", default="fp_31415"
    )


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
                    break
                try:
                    data = json.loads(data)
                    chunk = MinerResponse(**data)
                    if not chunk.choices[0].delta.content:
                        continue
                    self.streaming_chunks.append(chunk)
                    yield chunk
                except Exception as e:
                    logger.error("Error", e)
                    logger.error("Failed chunk:", data)
                    continue  # Continue instead of break to handle invalid chunks

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
            "streaming_chunks": self.streaming_chunks,
            "miner_payload": self.miner_payload,
        }

    def verify(self) -> bool:
        if len(self.streaming_chunks) == 0:
            return False
        if not len(self.completion):
            return False
        return True

    def to_headers(self) -> dict:
        # Get base headers from parent class
        headers = super().to_headers()

        # Add model information from miner_payload if available
        if self.miner_payload and self.miner_payload.model:
            headers["bt_header_input_obj_miner_payload"] = base64.b64encode(
                json.dumps(
                    {
                        "model": self.miner_payload.model,
                    }
                ).encode("utf-8")
            ).decode()

        return headers


class ImagePrompt(BaseModel):
    """
    Dall-E 3 prompt with parameters
    - <prompt> --ar 16:9 --q hd --style natural
    This class parses image generation prompts with optional parameters
    """

    prompt: str = Field(description="The main prompt text for image generation")
    size: str = Field(
        description="Size: [1024x1024, 1792x1024, 1024x1792]", default="1024x1024"
    )
    quality: str = Field(
        description="Image quality setting (hd, standard)", default="hd"
    )
    style: str = Field(description="Visual style modifier", default="natural")

    model: str = Field(
        description="Model to use for image generation", default="dall-e-3"
    )

    @validator("prompt")
    def validate_prompt(cls, v):
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v.strip()

    @validator("size")
    def validate_size(cls, v):
        valid_sizes = ["1024x1024", "1792x1024", "1024x1792"]
        if v not in valid_sizes:
            raise ValueError(f"Invalid size. Must be one of: {valid_sizes}")
        return v

    @validator("quality")
    def validate_quality(cls, v):
        valid_qualities = ["hd", "standard"]
        if v not in valid_qualities:
            raise ValueError(f"Invalid quality. Must be one of: {valid_qualities}")
        return v

    def to_string(self) -> str:
        """Convert the prompt and parameters to a command-line style string"""
        return f"{self.prompt} --ar {self.size} --q {self.quality} --style {self.style} --model {self.model}"

    @classmethod
    def from_string(cls, prompt_str: str) -> "ImagePrompt":
        """Parse a command-line style string into an ImagePrompt instance"""
        parts = prompt_str.split()
        params = {
            "prompt": [],
            "size": "1024x1024",
            "quality": "hd",
            "style": "natural",
            "model": "dall-e-3",
        }

        i = 0
        while i < len(parts):
            if parts[i].startswith("--"):
                param = parts[i][2:]
                if i + 1 >= len(parts):
                    raise ValueError(f"Missing value for parameter: {param}")

                if param == "ar":
                    params["size"] = parts[i + 1]
                elif param == "q":
                    params["quality"] = parts[i + 1]
                elif param == "style":
                    params["style"] = parts[i + 1]
                elif param == "model":
                    params["model"] = parts[i + 1]
                i += 2
            else:
                params["prompt"].append(parts[i])
                i += 1

        params["prompt"] = " ".join(params["prompt"])
        return cls(**params)


def mimic_chat_completion_chunk(
    content: str,
    id: str = "",
    model: str = "",
    created: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
) -> ChatCompletionChunk:
    data = {
        "id": id,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "logprobs": {},
                "finish_reason": "stop",
            }
        ],
        "created": created,
        "model": model,
        "object": "chat.completion.chunk",
        "system_fingerprint": "fp_31415",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }
    return ChatCompletionChunk(**data)
