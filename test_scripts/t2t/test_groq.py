import asyncio
import os
import traceback
from groq import AsyncGroq

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("API key not found in environment variables")

groq_client = AsyncGroq()
groq_client.api_key = api_key

async def send_groq_request(prompt, model="gemma-7b-it"):
    try:
        stream = await groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            model=model,
            seed=1234,
            temperature=0.0001,
            max_tokens=4096,
        )
        collected_messages = []

        async for part in stream:
            print(part.choices[0].delta.content or "")
            collected_messages.append(part.choices[0].delta.content or "")

        all_messages = ''.join(collected_messages)
        return all_messages

    except Exception as e:
        print(f"Got exception when calling groq {e}")
        traceback.print_exc()
        return "Error calling model"

async def main():
    prompts = ["count to 10", "tell me a joke"]
    tasks = [send_groq_request(prompt) for prompt in prompts]

    responses = await asyncio.gather(*tasks)
    for response in responses:
        print(response)

asyncio.run(main())
