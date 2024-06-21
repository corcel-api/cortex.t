import asyncio
import os
import traceback
from openai import OpenAI
from openai import AsyncOpenAI

OpenAI.api_key = os.environ.get('OPENAI_API_KEY')
if not OpenAI.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = AsyncOpenAI(timeout=30)

async def send_openai_request(message, engine="gpt-4o"):
    filtered_messages = [
        {
        "role": message["role"],
        "content": [],
        }
    ]
    if message.get("content"):
        filtered_messages[0]["content"].append(
            {
                "type": "text",
                "text": message["content"],
            }
        )
    if message.get("image"):
        image_url = message.get("image")
        filtered_messages[0]["content"].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            }
        )
    response = await client.chat.completions.create(
        messages=filtered_messages,
        stream=True,
        model=engine,
        seed=1234,
        temperature=0.0001,
        max_tokens=2000,
    )
    buffer = []
    async for chunk in response:
        token = chunk.choices[0].delta.content or ""
        buffer.append(token)
        print(token)
    joined_buffer = ''.join(buffer)
    return joined_buffer


async def main():
    messages = [
        {"role": "user", "content": "count to 10"},
        {"role": "user", "content": "what do you see here?", "image": "https://cdn.britannica.com/80/140480-131-28E57753/Dromedary-camels.jpg"}
    ]
    tasks = [send_openai_request(message) for message in messages]

    responses = await asyncio.gather(*tasks)
    for response in responses:
        print(response)

asyncio.run(main())
