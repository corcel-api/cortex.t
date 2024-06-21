from anthropic_bedrock import AsyncAnthropicBedrock, HUMAN_PROMPT, AI_PROMPT, AnthropicBedrock

AsyncClient = AsyncAnthropicBedrock(
    # default is 10 minutes
    # more granular timeout options:  timeout=httpx.Timeout(60.0, read=5.0, write=10.0, connect=2.0),
    timeout=60.0,
)
client = AnthropicBedrock()

question = """
Hey Claude! How can I recursively list all files in a directory in Python?
"""
models = ["anthropic.claude-v2:1", "anthropic.claude-instant-v1", "anthropic.claude-v1", "anthropic.claude-v2", "anthropic.claude-3-sonnet-20240229-v1:0"]

# Define an async function
async def run_async_code():

    stream = await AsyncClient.completions.create(
    prompt=f"{HUMAN_PROMPT} {question} {AI_PROMPT}",
    max_tokens_to_sample=300,
    temperature=0.01, # must be <= 1.0
    top_p=1,
    model=models[-1],
    stream=True,
    )

    async for completion in stream:
        print(completion.completion, end="", flush=True)
    print("\n")

# Run the async function in an event loop
if __name__ == "__main__":
    import asyncio
    asyncio.run(run_async_code())



# streaming with non async client

print("------ streamed response ------")
stream = client.completions.create(
    model=models[1],
    prompt=f"{HUMAN_PROMPT} {question}{AI_PROMPT}",
    max_tokens_to_sample=500,
    stream=True,
)
for item in stream:
    print(item.completion, end="")
print()
