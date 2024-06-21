from anthropic import AsyncAnthropic
import os
import base64
import httpx
import asyncio
from cortext.utils import get_api_key, generate_messages_to_claude


anthropic_key = get_api_key("Anthropic", "ANTHROPIC_API_KEY")
anthropic_client = AsyncAnthropic()
anthropic_client.api_key = anthropic_key

messages = [
    {
        "role": "system",
        "content": "respond in spanish"
    },
    {
        "role": "user",
        "image": "https://cdn.britannica.com/80/140480-131-28E57753/Dromedary-camels.jpg",
        "content": "how many animals on this picture?"
    }
]
max_tokens = 100
model = "claude-3-opus-20240229"


async def generate_messages_to_claude(messages):
            system_prompt = None
            filtered_messages = []
            for message in messages:
                if message["role"] == "system":
                    system_prompt = message["content"]
                else:
                    message_to_append = {
                            "role": message["role"],
                            "content": [],
                        }
                    if message.get("image"):
                        image_url = message.get("image")
                        image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
                        message_to_append["content"].append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_data,
                                },
                            }
                        )
                    if message.get("content"):
                        message_to_append["content"].append(
                            {
                                "type": "text",
                                "text": message["content"],
                            }
                        )
                    filtered_messages.append(message_to_append)
            return filtered_messages, system_prompt


async def call_anthropic(messages, max_tokens, model):
    filtered_messages, system_prompt = await generate_messages_to_claude(messages)

    stream_kwargs = {
        "max_tokens": max_tokens,
        "messages": filtered_messages,
        "model": model,
    }

    if system_prompt:
        stream_kwargs["system"] = system_prompt

    completion = anthropic_client.messages.stream(**stream_kwargs)
    async with completion as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)

    # Send final message to close the stream
    print("\n")

async def main():
    await call_anthropic(messages, max_tokens, model)

if __name__ == "__main__":
    asyncio.run(main())
