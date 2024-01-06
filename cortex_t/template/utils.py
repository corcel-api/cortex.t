from __future__ import annotations

import ast
import asyncio
import json
import base64
import math
import os
import random
import re
import traceback
from typing import Optional
from stability_sdk import client as stability_client

import bittensor as bt
import requests
import wandb

from .. import template

from . import client

list_update_lock = asyncio.Lock()
import anthropic_bedrock
from anthropic_bedrock import AsyncAnthropicBedrock

# Set up the API connection
stability_api = stability_client.StabilityInference(
    key=os.environ['STABILITY_KEY'],
    verbose=True,
    engine="stable-diffusion-xl-1024-v1-0"
)


def load_state_from_file(filename: str = "state.json"):
    if os.path.exists(filename):
        with open(filename, "r") as file:
            bt.logging.debug("loaded previous state")
            return json.load(file)
    else:
        bt.logging.debug("initialized new global state")
        return {
            "text": {"themes": None, "questions": None, "theme_counter": 0, "question_counter": 0},
            "images": {"themes": None, "questions": None, "theme_counter": 0, "question_counter": 0}
        }

state = load_state_from_file()


def get_state():
    global state
    if state is None:
        load_state_from_file()
    return state


def save_state_to_file(state, filename="state.json"):
    with open(filename, "w") as file:
        bt.logging.success(f"saved global state to {filename}")
        json.dump(state, file)


def get_validators_with_runs_in_all_projects():
    api = wandb.Api()
    validators_runs = {project: set() for project in projects}

    # Retrieve runs for each project and store validator UIDs
    for project in template.PROJECT_NAMES:
        runs = api.runs(f"cortex-t/{project}")
        for run in runs:
            if run.config['type'] == 'validator':
                validators_runs[project].add(run.config['uid'])

    # Find common validators across all projects
    common_validators = set.intersection(*validators_runs.values())
    return common_validators


async def get_list(list_type, num_questions_needed, theme=None):
    prompts_in_question = {'text_questions': 10, 'images_questions': 20}
    list_type_mapping = {
        "text_questions": {
            "default": template.INSTRUCT_DEFAULT_QUESTIONS,
            "prompt": "placeholder"
        },
        "images_questions": {
            "default": template.IMAGE_DEFAULT_QUESTIONS,
            "prompt": f"Provide a python-formatted list of {prompts_in_question[list_type]} creative and detailed scenarios for image generation, each inspired by the theme '{theme}'. The scenarios should be diverse, thoughtful, and possibly out-of-the-box interpretations related to '{theme}'. Each element in the list should be a concise, but a vividly descriptive situation designed to inspire visually rich stories. Format these elements as comma-separated, quote-encapsulated strings in a single Python list."
        }
    }

    selected_prompts = []
    if list_type == "text_questions":
        question_pool = []
        for complexity_level in range(1, 21):
            for relevance_level in range(1, 21):
                prompt = (f"Generate a python-formatted list of {prompts_in_question[list_type]} questions "
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

    bt.logging.debug(f"num_questions_needed: {num_questions_needed}, "
                     f"list_type: {list_type}, selected_prompts: {selected_prompts}")

    tasks = [
        call_openai([{'role': "user", 'content': prompt}], 0.65, "gpt-3.5-turbo", random.randint(1, 10000))
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
                extracted_lists += extracted_list
            else:
                # Retry logic for each prompt if needed
                for retry in range(max_retries):
                    try:
                        random_seed = random.randint(1, 10000)
                        messages = [{'role': "user", 'content': selected_prompts[i]}]
                        new_answer = await call_openai(messages, 0.85, "gpt-4-1106-preview", random_seed)
                        new_answer = new_answer.replace("\n", " ") if new_answer else ""
                        new_extracted_list = extract_python_list(new_answer)
                        if new_extracted_list:
                            extracted_lists += new_extracted_list
                            break
                        bt.logging.error(f"no list found in {new_answer}")
                    except Exception as e:
                        bt.logging.error(f"Exception on retry {retry + 1} for prompt '{selected_prompts[i]}': "
                                         f"{e}\n{traceback.format_exc()}")
        except Exception as e:
            bt.logging.error(f"Exception in processing initial response for prompt '{selected_prompts[i]}': "
                             f"{e}\n{traceback.format_exc()}")

    if not extracted_lists:
        bt.logging.error("No valid lists found after processing and retries, returning None")
        return None

    return extracted_lists


async def update_counters_and_get_new_list(category, item_type, num_questions_needed, theme=None):

    async def get_items(category, item_type, theme=None):
        if item_type == "themes":
            if category == "images":
                return template.IMAGE_THEMES
            return template.INSTRUCT_DEFAULT_THEMES
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

    list_type = f"{category}_{item_type}"

    async with list_update_lock:
        items = state[category][item_type]

        # Logging the current state before fetching new items
        bt.logging.debug(f"Queue for {list_type}: {len(items) if items else 0} items")

        # Fetch new items if the list is empty
        if not items:
            items = await get_items(category, item_type, theme)
            state[category][item_type] = items
            bt.logging.debug(f"Fetched new list for {list_type}, containing {len(items)} items")

        item = items.pop() if items else None
        if not items:
            state[category][item_type] = None

    return item


async def get_question(category, num_questions_needed):
    if category not in ["text", "images"]:
        raise ValueError("Invalid category. Must be 'text' or 'images'.")

    question = await update_counters_and_get_new_list(category, "questions", num_questions_needed)
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
        if processed_text[i] == '#':
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
            if char == '[':
                found_first_bracket = True
            cleaned_text.append(char)
            i += 1
            continue

        if char == '"':
            # Look for preceding comma or bracket, skipping spaces
            preceding_char_index = i - 1
            found_comma_or_bracket = False

            while preceding_char_index >= 0:
                if no_comments_text[preceding_char_index] in '[,':  # Check for comma or opening bracket
                    found_comma_or_bracket = True
                    break
                if no_comments_text[preceding_char_index] not in ' \n':  # Ignore spaces and new lines
                    break
                preceding_char_index -= 1

            following_char_index = i + 1
            while following_char_index < len(no_comments_text) and no_comments_text[following_char_index] in ' \n':
                following_char_index += 1

            if found_comma_or_bracket or \
               (following_char_index < len(no_comments_text) and no_comments_text[following_char_index] in '],'):
                inside_quotes = not inside_quotes
            else:
                i += 1
                continue  # Skip this quote

            cleaned_text.append(char)
            i += 1
            continue

        if char == ' ':
            # Skip spaces if not inside quotes and if the space is not between words
            if not inside_quotes and (i == 0 or no_comments_text[i - 1] in ' ,[' or no_comments_text[i + 1] in ' ,]'):
                i += 1
                continue

        cleaned_text.append(char)
        i += 1

    cleaned_str = ''.join(cleaned_text)
    cleaned_str = re.sub(r"\[\s+", "[", cleaned_str)
    cleaned_str = re.sub(r"\s+\]", "]", cleaned_str)
    cleaned_str = re.sub(r"\s*,\s*", ", ", cleaned_str)  # Ensure single space after commas

    start, end = cleaned_str.find('['), cleaned_str.rfind(']')
    if start != -1 and end != -1 and end > start:
        cleaned_str = cleaned_str[start:end + 1]

    return cleaned_str


def convert_to_list(text: str) -> list[str]:
    pattern = r'\d+\.\s'
    items = [item.strip() for item in re.split(pattern, text) if item]
    return items


def extract_python_list(text: str):
    try:
        if re.match(r'\d+\.\s', text):
            return convert_to_list(text)

        text = preprocess_string(text)
        bt.logging.trace(f"Postprocessed text = {text}")

        # Extracting list enclosed in square brackets
        match = re.search(r'\[((?:[^][]|"(?:\\.|[^"\\])*")*)\]', text, re.DOTALL)
        if match:
            list_str = match.group(1)

            # Using ast.literal_eval to safely evaluate the string as a list
            evaluated = ast.literal_eval('[' + list_str + ']')
            if isinstance(evaluated, list):
                return evaluated

    except Exception as e:
        bt.logging.error(f"found double quotes in list, trying again")

    return None


async def call_openai(messages, temperature, model, seed=1234, max_tokens=2048, top_p=1) -> str:
    for _ in range(2):
        bt.logging.debug(f"Calling Openai. Temperature = {temperature}, Model = {model}, Seed = {seed},  Messages = {messages}")
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
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

    return None



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

async def call_anthropic(prompt, temperature, model, max_tokens=2048, top_p=1, top_k=10000):
    try:
        client = AsyncAnthropicBedrock()
        bt.logging.debug(f"Calling Anthropic. Model = {model}, Prompt = {prompt}, Temperature = {temperature}, Max Tokens = {max_tokens}")
        completion = await client.completions.create(
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
        bt.logging.error(f"Error when calling Anthropic: {traceback.format_exc()}")
        await asyncio.sleep(0.5)

async def call_stability(prompt, seed, steps, cfg_scale, width, height, samples, sampler):
    bt.logging.debug(f"calling stability for {prompt, seed, steps, cfg_scale, width, height, samples, sampler}")

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
    url = "https://api.github.com/repos/corcel-api/cortex.t/contents/template/__init__.py"
    response = requests.get(url, timeout=10)
    if not response.ok:
        bt.logging.error("github api call failed")
        return None

    content = response.json()['content']
    decoded_content = base64.b64decode(content).decode('utf-8')
    lines = decoded_content.split('\n')
    if line_number > len(lines):
        raise Exception("Line number exceeds file length")

    version_line = lines[line_number - 1]
    version_match = re.search(r'__version__ = "(.*?)"', version_line)
    if not version_match:
        raise Exception("Version information not found in the specified line")

    return version_match.group(1)


def send_discord_alert(message, webhook_url):
    data = {
        "content": f"@everyone {message}",
        "username": "Subnet18 Updates"
    }
    try:
        response = requests.post(webhook_url, json=data, timeout=10)
        if response.status_code == 204:
            print("Discord alert sent successfully!")
        else:
            print(f"Failed to send Discord alert. Status code: {response.status_code}")
    except Exception as e:
        print(f"Failed to send Discord alert: {e}", exc_info=True)
