import asyncio
import os
import json
import aioboto3
import traceback
import time

bedrock_client_parameters = {
    "service_name": 'bedrock-runtime',
    "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY"),
    "aws_secret_access_key": os.environ.get("AWS_SECRET_KEY"),
    "region_name": "us-east-1"
}

models = [
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "cohere.command-r-v1:0", "meta.llama2-70b-chat-v1",
    "amazon.titan-text-express-v1",
    "mistral.mistral-7b-instruct-v0:2",
    "ai21.j2-mid-v1"
    ]


async def send_bedrock_request(message, model, max_tokens=100):
    async def generate_messages_to_claude(message):
        system_prompt = None
        filtered_messages = []
        message_to_append = {
                "role": message["role"],
                "content": [],
            }
        if message.get("content"):
            message_to_append["content"].append(
                {
                    "type": "text",
                    "text": message["content"],
                }
            )
        filtered_messages.append(message_to_append)
        return filtered_messages, system_prompt

    async def generate_request(message):
        if model.startswith("cohere"):
            native_request = {
                "message": message["content"],
                "max_tokens": max_tokens,
            }
        elif model.startswith("meta"):
            native_request = {
                "prompt": message["content"],
                "max_gen_len": max_tokens,
            }
        elif model.startswith("anthropic"):
            message, system_prompt = await generate_messages_to_claude(message)
            native_request = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": message,
                "max_tokens": max_tokens,
            }
            if system_prompt:
                native_request["system"] = system_prompt
        elif model.startswith("mistral"):
            native_request = {
                "prompt": message["content"],
                "max_tokens": max_tokens,
            }
        elif model.startswith("amazon"):
            native_request = {
                "inputText": message["content"],
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                },
            }
        elif model.startswith("ai21"):
            native_request = {
                "prompt": message["content"],
                "maxTokens": max_tokens,
            }
        request = json.dumps(native_request)
        return request

    async def extract_token(chunk):
        if model.startswith("cohere"):
            token = chunk.get("text") or ""
        elif model.startswith("meta"):
            token = chunk.get("generation") or ""
        elif model.startswith("anthropic"):
            token = ""
            if chunk['type'] == 'content_block_delta':
                if chunk['delta']['type'] == 'text_delta':
                    token = chunk['delta']['text']
        elif model.startswith("mistral"):
            token = chunk.get("outputs")[0]["text"] or ""
        elif model.startswith("amazon"):
            token = chunk.get("outputText") or ""
        elif model.startswith("ai21"):
            token = json.loads(message)["completions"][0]["data"]["text"]
        return token
    aws_session = aioboto3.Session()
    aws_bedrock_client = aws_session.client(**bedrock_client_parameters)

    request = await generate_request(message)
    async with aws_bedrock_client as client:
        if model.startswith("ai21"):
            response = await client.invoke_model(
                modelId=model, body=request
            )
            message = await response['body'].read()
            message = await extract_token(message)
            print("-------------\n" + model + " response: \n")
            print(message)
        else:
            stream = await client.invoke_model_with_response_stream(
                modelId=model, body=request
            )

            buffer = []
            n = 1
            async for event in stream["body"]:
                try:
                    chunk = json.loads(event["chunk"]["bytes"])
                    token = await extract_token(chunk)
                    buffer.append(token)
                    if len(buffer) == n:
                        joined_buffer = "".join(buffer)
                except:
                    print(traceback.format_exc())

            if buffer:
                joined_buffer = "".join(buffer)
            print("-------------\n" + model + " response: \n")
            print(joined_buffer)


async def main():

    messages = [{"role": "user", "content": "count to 10"}]
    tasks = []
    for message in messages:
        for model in models:
            print("model:", model)
            await send_bedrock_request(message, model)
            # tasks.append(send_bedrock_request(message, model))
            await asyncio.sleep(1)

    # await asyncio.gather(*tasks)

asyncio.run(main())
