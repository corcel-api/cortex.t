from __future__ import annotations

import aioboto3
import ast
import asyncio
import base64
import io
import json
import math
import httpx
import os
import random
import re
import traceback
from typing import Any, Optional

import anthropic
import anthropic_bedrock
import bittensor as bt
import boto3
import cortext
import google.generativeai as genai
import requests
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import wandb
from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic, AsyncAnthropic
from anthropic_bedrock import AsyncAnthropicBedrock
from groq import AsyncGroq
from PIL import Image
from stability_sdk import client as stability_client
from cortext import IMAGE_PROMPTS

from . import client

list_update_lock = asyncio.Lock()

# Function to get API key from environment variables
def get_api_key(service_name, env_var):
    key = os.environ.get(env_var)
    if not key:
        raise ValueError(
            f"{service_name} API key not found in environment variables. "
            f"Go to the respective service's settings to get one. Then set it as {env_var} in your .env"
        )
    return key

pixabay_key = get_api_key("Pixabay", "PIXABAY_API_KEY")

# Stability API
# stability_key = get_api_key("Stability", "STABILITY_API_KEY")
# stability_api = stability_client.StabilityInference(key=stability_key, verbose=True)

# Anthropic
anthropic_key = get_api_key("Anthropic", "ANTHROPIC_API_KEY")
anthropic_client = AsyncAnthropic()
anthropic_client.api_key = anthropic_key

# Google
google_key=get_api_key("Google", "GOOGLE_API_KEY")
genai.configure(api_key=google_key)

# Anthropic Bedrock
anthropic_bedrock_client = AsyncAnthropicBedrock()

# Groq
groq_key = get_api_key("Groq", "GROQ_API_KEY")
groq_client = AsyncGroq()
groq_client.api_key = groq_key

# AWS Bedrock
bedrock_client_parameters = {
    "service_name": 'bedrock-runtime',
    "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY"),
    "aws_secret_access_key": os.environ.get("AWS_SECRET_KEY"),
    "region_name": "us-east-1"
}


def validate_state(data):
    expected_structure = {
        "text": {"themes": list, "questions": list, "theme_counter": int, "question_counter": int},
        "images": {"themes": list, "questions": list, "theme_counter": int, "question_counter": int},
    }

    def check_subdict(subdict, expected):
        if not isinstance(subdict, dict):
            return False
        for key, expected_type in expected.items():
            if key not in subdict or not isinstance(subdict[key], expected_type):
                return False
        return True

    def check_list_of_dicts(lst):
        if not isinstance(lst, list):
            return False
        for item in lst:
            if not isinstance(item, dict):
                return False
        return True

    if not isinstance(data, dict):
        return False
    for key, expected_subdict in expected_structure.items():
        if key not in data or not check_subdict(data[key], expected_subdict):
            return False
        if key == "text" and not check_list_of_dicts(data[key]["questions"]):
            return False

    return True


def load_state_from_file(filename: str):
    load_success = False
    state_is_valid = False

    # Check if the file exists
    if os.path.exists(filename):
        with open(filename, "r") as file:
            try:
                # Attempt to load JSON from the file
                bt.logging.debug("loaded previous state")
                state = json.load(file)
                state_is_valid = validate_state(state)
                if not state_is_valid:
                    raise Exception("State is invalid")
                load_success = True  # Set flag to true as the operation was successful
                return state
            except Exception as e:  # Catch specific exceptions for better error handling
                bt.logging.error(f"error loading state, deleting and resetting it. Error: {e}")
                os.remove(filename)  # Delete if error

    # If the file does not exist or there was an error
    if not load_success or not state_is_valid:
        bt.logging.debug("initialized new global state")
        # Return the default state structure
        return {
            "text": {"themes": [], "questions": [], "theme_counter": 0, "question_counter": 0},
            "images": {"themes": [], "questions": [], "theme_counter": 0, "question_counter": 0},
        }


state = None


def get_state(path):
    global state
    if not state:
        state = load_state_from_file(path)
    return state


def save_state_to_file(state, filename="state.json"):
    with open(filename, "w") as file:
        bt.logging.success(f"saved global state to {filename}")
        json.dump(state, file)


def get_validators_with_runs_in_all_projects():
    api = wandb.Api()
    validators_runs = {project: set() for project in projects}

    # Retrieve runs for each project and store validator UIDs
    for project in cortext.PROJECT_NAMES:
        runs = api.runs(f"cortex-t/{project}")
        for run in runs:
            if run.config["type"] == "validator":
                validators_runs[project].add(run.config["uid"])

    # Find common validators across all projects
    common_validators = set.intersection(*validators_runs.values())
    return common_validators


def fetch_random_image_urls(num_images):
    try:
        url = f"https://pixabay.com/api/?key={pixabay_key}&per_page={num_images}&order=popular"
        response = requests.get(url)
        response.raise_for_status()
        images = response.json().get('hits', [])
        return [image['webformatURL'] for image in images]
    except Exception as e:
        print(f"Error fetching random images: {e}")
        return []


async def get_list(list_type, num_questions_needed, theme=None):
    prompts_in_question = {"text_questions": 10, "images_questions": 20}
    list_type_mapping = {
        "text_questions": {"default": cortext.INSTRUCT_DEFAULT_QUESTIONS, "prompt": "placeholder"},
        "images_questions": {
            "default": cortext.IMAGE_DEFAULT_QUESTIONS,
            "prompt": f"Provide a python-formatted list of {prompts_in_question[list_type]} creative and detailed scenarios for image generation, each inspired by the theme '{theme}'. The scenarios should be diverse, thoughtful, and possibly out-of-the-box interpretations related to '{theme}'. Each element in the list should be a concise, but a vividly descriptive situation designed to inspire visually rich stories. Format these elements as comma-separated, quote-encapsulated strings in a single Python list.",
        },
    }

    selected_prompts = []
    if list_type == "text_questions":
        question_pool = []
        for complexity_level in range(1, 21):
            for relevance_level in range(1, 21):
                random_int = random.randint(1, 100)
                if random_int <= 50:
                    task_type = "questions"
                else:
                    task_type = random.sample(cortext.INSTRUCT_TASK_OPTIONS, 1)

                prompt = (f"Generate a python-formatted list of {prompts_in_question[list_type]} {task_type} "
                          f"or instruct tasks related to the theme '{theme}', each with a complexity level "
                          f"of {complexity_level} out of 20 and a relevance level to the theme "
                          f"of {relevance_level} out of 20. These tasks should varyingly explore "
                          f"{theme} in a manner that is consistent with their assigned complexity and relevance "
                          f"levels to the theme, allowing for a diverse and insightful engagement about {theme}. "
                          f"Format the questions as comma-separated, quote-encapsulated strings "
                          f"in a single Python list.")
                question_pool.append(prompt)

        random.shuffle(question_pool)
        num_questions_to_select = min(
            math.ceil(num_questions_needed / prompts_in_question[list_type]),
            len(question_pool),
        )
        selected_prompts = random.sample(question_pool, num_questions_to_select)
    else:
        num_questions_to_select = math.ceil(num_questions_needed / prompts_in_question[list_type])
        selected_prompts = [list_type_mapping[list_type]["prompt"]] * num_questions_to_select

    bt.logging.debug(
        f"num_questions_needed: {num_questions_needed}, "
        f"list_type: {list_type}, selected_prompts: {selected_prompts}"
    )

    tasks = [
        call_openai([{'role': "user", 'content': prompt}], 0.65, "gpt-4o", random.randint(1, 10000))
        for prompt in selected_prompts
    ]

    responses = await asyncio.gather(*tasks)
    extracted_lists = []
    max_retries = 5
    for i, answer in enumerate(responses):
        try:
            answer = answer.replace("\n", " ") if answer else ""
            extracted_list = extract_python_list(answer)
            if extracted_list:
                if list_type == "text_questions":
                    extracted_lists +=  [{"prompt": s} for s in extracted_list]
                else:
                    extracted_lists += extracted_list
            else:
                # Retry logic for each prompt if needed
                for retry in range(max_retries):
                    try:
                        random_seed = random.randint(1, 10000)
                        messages = [{"role": "user", "content": selected_prompts[i]}]
                        new_answer = await call_openai(messages, 0.85, "gpt-4-0125-preview", random_seed)
                        new_answer = new_answer.replace("\n", " ") if new_answer else ""
                        new_extracted_list = extract_python_list(new_answer)
                        if new_extracted_list:
                            extracted_lists += {"prompt": new_extracted_list}
                            break
                        bt.logging.error(f"no list found in {new_answer}")
                    except Exception as e:
                        bt.logging.error(
                            f"Exception on retry {retry + 1} for prompt '{selected_prompts[i]}': "
                            f"{e}\n{traceback.format_exc()}"
                        )
        except Exception as e:
            bt.logging.error(
                f"Exception in processing initial response for prompt '{selected_prompts[i]}': "
                f"{e}\n{traceback.format_exc()}"
            )

    if not extracted_lists:
        bt.logging.error("No valid lists found after processing and retries, returning None")
        return None

    if list_type == "text_questions":
        images_from_pixabay = fetch_random_image_urls(prompts_in_question[list_type])
        for image_url in images_from_pixabay:
            extracted_lists.append(
                {
                    "prompt": random.choice(IMAGE_PROMPTS),
                    "image": image_url,
                }
            )

    return extracted_lists


async def update_counters_and_get_new_list(category, item_type, num_questions_needed, vision, theme=None):

    async def get_items(category, item_type, theme=None):
        if item_type == "themes":
            if category == "images":
                return cortext.IMAGE_THEMES
            return cortext.INSTRUCT_DEFAULT_THEMES
        else:
            # Never fail here, retry until valid list is found
            while True:
                theme = await get_random_theme(category)
                if theme is not None:
                    return await get_list(f"{category}_questions", num_questions_needed, theme)

    async def get_random_theme(category):
        themes = state[category]["themes"]
        if not themes:
            themes = await get_items(category, "themes")
            state[category]["themes"] = themes
        return random.choice(themes)

    async def get_item_from_list(items, vision):
        if vision:
            return items.pop() if items else None
        else:
            for i, itm in enumerate(items):
                if 'image' not in itm:
                    return items.pop(i)
            return None

    list_type = f"{category}_{item_type}"

    async with list_update_lock:
        items = state[category][item_type]

        bt.logging.debug(f"Queue for {list_type}: {len(items) if items else 0} items")

        item = await get_item_from_list(items, vision)

        if not item:
            bt.logging.info(f"Item not founded in items: {items}. Calling get_items!")
            items = await get_items(category, item_type, theme)
            bt.logging.info(f"Items generated: {items}")
            state[category][item_type] = items
            bt.logging.debug(f"Fetched new list for {list_type}, containing {len(items)} items")

            item = await get_item_from_list(items, vision)

        if not items:
            state[category][item_type] = []

    return item


async def get_question(category, num_questions_needed, vision=False):
    if category not in ["text", "images"]:
        raise ValueError("Invalid category. Must be 'text' or 'images'.")

    question = await update_counters_and_get_new_list(category, "questions", num_questions_needed, vision)
    return question


def preprocess_string(text: str) -> str:
    processed_text = text.replace("\t", "")
    placeholder = "___SINGLE_QUOTE___"
    processed_text = re.sub(r"(?<=\w)'(?=\w)", placeholder, processed_text)
    processed_text = processed_text.replace("'", '"').replace(placeholder, "'")

    # First, remove all comments, ending at the next quote
    no_comments_text = ""
    i = 0
    in_comment = False
    while i < len(processed_text):
        if processed_text[i] == "#":
            in_comment = True
        elif processed_text[i] == '"' and in_comment:
            in_comment = False
            no_comments_text += processed_text[i]  # Keep the quote that ends the comment
            i += 1
            continue
        if not in_comment:
            no_comments_text += processed_text[i]
        i += 1

    # Now process the text without comments for quotes
    cleaned_text = []
    inside_quotes = False
    found_first_bracket = False

    i = 0
    while i < len(no_comments_text):
        char = no_comments_text[i]

        if not found_first_bracket:
            if char == "[":
                found_first_bracket = True
            cleaned_text.append(char)
            i += 1
            continue

        if char == '"':
            # Look for preceding comma or bracket, skipping spaces
            preceding_char_index = i - 1
            found_comma_or_bracket = False

            while preceding_char_index >= 0:
                if no_comments_text[preceding_char_index] in "[,":  # Check for comma or opening bracket
                    found_comma_or_bracket = True
                    break
                if no_comments_text[preceding_char_index] not in " \n":  # Ignore spaces and new lines
                    break
                preceding_char_index -= 1

            following_char_index = i + 1
            while following_char_index < len(no_comments_text) and no_comments_text[following_char_index] in " \n":
                following_char_index += 1

            if found_comma_or_bracket or (
                following_char_index < len(no_comments_text) and no_comments_text[following_char_index] in "],"
            ):
                inside_quotes = not inside_quotes
            else:
                i += 1
                continue  # Skip this quote

            cleaned_text.append(char)
            i += 1
            continue

        if char == " ":
            # Skip spaces if not inside quotes and if the space is not between words
            if not inside_quotes and (i == 0 or no_comments_text[i - 1] in " ,[" or no_comments_text[i + 1] in " ,]"):
                i += 1
                continue

        cleaned_text.append(char)
        i += 1

    cleaned_str = "".join(cleaned_text)
    cleaned_str = re.sub(r"\[\s+", "[", cleaned_str)
    cleaned_str = re.sub(r"\s+\]", "]", cleaned_str)
    cleaned_str = re.sub(r"\s*,\s*", ", ", cleaned_str)  # Ensure single space after commas

    start, end = cleaned_str.find("["), cleaned_str.rfind("]")
    if start != -1 and end != -1 and end > start:
        cleaned_str = cleaned_str[start : end + 1]

    return cleaned_str


def convert_to_list(text: str) -> list[str]:
    pattern = r"\d+\.\s"
    items = [item.strip() for item in re.split(pattern, text) if item]
    return items


def extract_python_list(text: str):
    try:
        if re.match(r"\d+\.\s", text):
            return convert_to_list(text)

        text = preprocess_string(text)
        bt.logging.trace(f"Postprocessed text = {text}")

        # Extracting list enclosed in square brackets
        match = re.search(r'\[((?:[^][]|"(?:\\.|[^"\\])*")*)\]', text, re.DOTALL)
        if match:
            list_str = match.group(1)

            # Using ast.literal_eval to safely evaluate the string as a list
            evaluated = ast.literal_eval("[" + list_str + "]")
            if isinstance(evaluated, list):
                return evaluated

    except Exception as e:
        bt.logging.error(f"found double quotes in list, trying again")

    return None


async def call_openai(messages, temperature, model, seed=1234, max_tokens=2048, top_p=1):
    for _ in range(2):
        bt.logging.debug(
            f"Calling Openai. Temperature = {temperature}, Model = {model}, Seed = {seed},  Messages = {messages}"
        )
        try:
            message = messages[0]
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
                model=model,
                messages=filtered_messages,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            response = response.choices[0].message.content
            bt.logging.trace(f"validator response is {response}")
            return response

        except Exception as e:
            bt.logging.error(f"Error when calling OpenAI: {traceback.format_exc()}")
            await asyncio.sleep(0.5)


async def call_gemini(messages, temperature, model, max_tokens, top_p, top_k):
    print(f"Calling Gemini. Temperature = {temperature}, Model = {model}, Messages = {messages}")
    try:
        model = genai.GenerativeModel(model)
        response = model.generate_content(
            str(messages),
            stream=False,
            generation_config=genai.types.GenerationConfig(
                # candidate_count=1,
                # stop_sequences=['x'],
                temperature=temperature,
                # max_output_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                # seed=seed,
            ),
        )

        print(f"validator response is {response.text}")
        return response.text
    except:
        print(f"error in call_gemini {traceback.format_exc()}")


# anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# async def call_anthropic(prompt, temperature, model, max_tokens=2048, top_p=1, top_k=10000) -> str:

#     for _ in range(2):
#         bt.logging.debug(f"Calling Anthropic. Model = {model}, Prompt = {prompt}")
#         try:
#             completion = anthropic.completions.create(
#                 model=model,
#                 max_tokens_to_sample=max_tokens,
#                 prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
#                 temperature=temperature,
#                 top_p=top_p,
#                 top_k=top_k,
#             )
#             response = completion.completion
#             bt.logging.debug(f"Validator response is {response}")
#             return response

#         except Exception as e:
#             bt.logging.error(f"Error when calling Anthropic: {traceback.format_exc()}")
#             await asyncio.sleep(0.5)

#     return None


async def call_anthropic_bedrock(prompt, temperature, model, max_tokens=2048, top_p=1, top_k=10000):
    try:
        bt.logging.debug(
            f"Calling Bedrock via Anthropic. Model = {model}, Prompt = {prompt}, Temperature = {temperature}, Max Tokens = {max_tokens}"
        )
        completion = await anthropic_bedrock_client.completions.create(
            model=model,
            max_tokens_to_sample=max_tokens,
            temperature=temperature,
            prompt=f"{anthropic_bedrock.HUMAN_PROMPT} {prompt} {anthropic_bedrock.AI_PROMPT}",
            top_p=top_p,
            top_k=top_k,
        )
        bt.logging.trace(f"Validator response is {completion.completion}")

        return completion.completion
    except Exception as e:
        bt.logging.error(f"Error when calling Bedrock via Anthropic: {traceback.format_exc()}")
        await asyncio.sleep(0.5)

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

async def call_anthropic(messages, temperature, model, max_tokens, top_p, top_k):
    try:
        bt.logging.info(
            f"calling Anthropic for {messages} with temperature: {temperature}, model: {model}, max_tokens: {max_tokens}, top_p: {top_p}, top_k: {top_k}"
        )
        filtered_messages, system_prompt = await generate_messages_to_claude(messages)

        kwargs = {
            "max_tokens": max_tokens,
            "messages": filtered_messages,
            "model": model,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        message = await anthropic_client.messages.create(**kwargs)
        bt.logging.debug(f"validator response is {message.content[0].text}")
        return message.content[0].text
    except:
        bt.logging.error(f"error in call_anthropic {traceback.format_exc()}")


async def call_groq(messages, temperature, model, max_tokens, top_p, seed):
    try:
        bt.logging.info(
            f"calling groq for {messages} with temperature: {temperature}, model: {model}, max_tokens: {max_tokens}, top_p: {top_p}"
        )

        kwargs = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "seed": seed,
        }

        message = await groq_client.chat.completions.create(**kwargs)
        bt.logging.debug(f"validator response is {message.choices[0].message.content}")
        return message.choices[0].message.content
    except:
        bt.logging.error(f"error in call_groq {traceback.format_exc()}")


async def call_bedrock(messages, temperature, model, max_tokens, top_p, seed):
    try:
        bt.logging.info(
            f"calling AWS Bedrock for {messages} with temperature: {temperature}, model: {model}, max_tokens: {max_tokens}, top_p: {top_p}"
        )

        async def generate_request():
            if model.startswith("cohere"):
                native_request = {
                    "message": messages[0]["content"],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "p": top_p,
                    "seed": seed,
                }
            elif model.startswith("meta"):
                native_request = {
                    "prompt": messages[0]["content"],
                    "temperature": temperature,
                    "max_gen_len": 2048 if max_tokens > 2048 else max_tokens,
                    "top_p": top_p,
                }
            elif model.startswith("anthropic"):
                message, system_prompt = await generate_messages_to_claude(messages)
                native_request = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": message,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                }
                if system_prompt:
                    native_request["system"] = system_prompt
            elif model.startswith("mistral"):
                native_request = {
                    "prompt": messages[0]["content"],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            elif model.startswith("amazon"):
                native_request = {
                    "inputText": messages[0]["content"],
                    "textGenerationConfig": {
                        "maxTokenCount": max_tokens,
                        "temperature": temperature,
                        "topP": top_p,
                    },
                }
            elif model.startswith("ai21"):
                native_request = {
                    "prompt": messages[0]["content"],
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                    "topP": top_p,
                }
            request = json.dumps(native_request)
            return request

        async def extract_message(message):
            if model.startswith("cohere"):
                message = json.loads(message)["text"]
            elif model.startswith("meta"):
                message = json.loads(message)["generation"]
            elif model.startswith("anthropic"):
                message = json.loads(message)["content"][0]["text"]
            elif model.startswith("mistral"):
                message = json.loads(message)["outputs"][0]["text"]
            elif model.startswith("amazon"):
                message = json.loads(message)["results"][0]["outputText"]
            elif model.startswith("ai21"):
                message = json.loads(message)["completions"][0]["data"]["text"]
            return message

        aws_session = aioboto3.Session()
        aws_bedrock_client = aws_session.client(**bedrock_client_parameters)

        async with aws_bedrock_client as client:
            request = await generate_request()
            response = await client.invoke_model(
                modelId=model, body=request
            )

            message = await response['body'].read()
            message = await extract_message(message)

        bt.logging.debug(f"validator response is {message}")
        return message
    except:
        bt.logging.error(f"error in call_bedrock {traceback.format_exc()}")


async def call_stability(prompt, seed, steps, cfg_scale, width, height, samples, sampler):
    # bt.logging.info(f"calling stability for {prompt, seed, steps, cfg_scale, width, height, samples, sampler}")
    bt.logging.info(f"calling stability for {prompt[:50]}...")

    # Run the synchronous stability_api.generate function in a separate thread
    meta = await asyncio.to_thread(
        stability_api.generate,
        prompt=prompt,
        seed=seed,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        samples=samples,
        # sampler=sampler,
    )

    # Convert image binary data to base64
    b64s = [base64.b64encode(artifact.binary).decode() for image in meta for artifact in image.artifacts]
    return b64s


# Github unauthorized rate limit of requests per hour is 60. Authorized is 5000.
def get_version(line_number: int = 22) -> Optional[str]:
    url = "https://api.github.com/repos/corcel-api/cortex.t/contents/cortext/__init__.py"
    response = requests.get(url, timeout=10)
    if not response.ok:
        bt.logging.error("github api call failed")
        return None

    content = response.json()["content"]
    decoded_content = base64.b64decode(content).decode("utf-8")
    lines = decoded_content.split("\n")
    if line_number > len(lines):
        raise Exception("Line number exceeds file length")

    version_line = lines[line_number - 1]
    version_match = re.search(r'__version__ = "(.*?)"', version_line)
    if not version_match:
        raise Exception("Version information not found in the specified line")

    return version_match.group(1)


def send_discord_alert(message, webhook_url):
    data = {"content": f"@everyone {message}", "username": "Subnet18 Updates"}
    try:
        response = requests.post(webhook_url, json=data, timeout=10)
        if response.status_code == 204:
            print("Discord alert sent successfully!")
        else:
            print(f"Failed to send Discord alert. Status code: {response.status_code}")
    except Exception as e:
        print(f"Failed to send Discord alert: {e}", exc_info=True)
