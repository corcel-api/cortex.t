import re
import ast
import math
import random
import asyncio
from cortex_t import template
import traceback
import bittensor as bt

from cortex_t.template import client


instruct_questions = []

def preprocess_string(text):
    processed_text = text.replace("\t", "").replace("\n", "")
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
                elif no_comments_text[preceding_char_index] not in ' \n':  # Ignore spaces and new lines
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

def convert_to_list(text):
    pattern = r'\d+\.\s'
    items = [item.strip() for item in re.split(pattern, text) if item]
    return items

def extract_python_list(text: str):
    try:
        if re.match(r'\d+\.\s', text):
            return convert_to_list(text)
        
        print(f"Preprocessed text = {text}")
        text = preprocess_string(text)
        print(f"Postprocessed text = {text}")

        # Extracting list enclosed in square brackets
        match = re.search(r'\[((?:[^][]|"(?:\\.|[^"\\])*")*)\]', text, re.DOTALL)
        if match:
            list_str = match.group(1)

            # Using ast.literal_eval to safely evaluate the string as a list
            evaluated = ast.literal_eval('[' + list_str + ']')
            if isinstance(evaluated, list):
                return evaluated

    except Exception as e:
        print(f"Unexpected error when extracting list: {e}\n{traceback.format_exc()}")

    return text


async def get_list(list_type, num_questions_needed, theme=None):
    prompts_in_question = {'text_questions': 10, 'images_questions': 20}
    list_type_mapping = {
        "text_themes": {
            "default": template.INSTRUCT_DEFAULT_THEMES,
            "prompt": "Please generate a list of broad, informational themes suitable for creating a diverse range of instructive prompts. These themes should be ideal for training an LLM to produce detailed, educational, and insightful content. They should span across multiple disciplines and be general enough to allow for the generation of many sub-topics. Each theme should be a seed for countless questions that delve into the specifics of the subject matter, aiming to inform and educate users about complex topics in an accessible way. Avoid themes that are too narrow or niche and focus on those that could be universally recognized and widely applicable for educational purposes."
        },
        "images_themes": {
            "default": template.IMAGE_DEFAULT_THEMES,
            "prompt": "Generate a Python list of 50 unique and broad creative themes for artistic inspiration. Each theme should be no more than four words, open to interpretation, and suitable for various artistic expressions. Present the list in a single-line Python list structure."
        },
        "text_questions": {
            "default": template.INSTRUCT_DEfAULT_QUESTIONS,
            "prompt": "placeholder"
        },
        "images_questions": {
            "default": template.IMAGE_DEFAULT_QUESTIONS,
            "prompt": f"Provide a Python list of {prompts_in_question[list_type]} creative and detailed scenarios for image generation, each inspired by the theme '{theme}'. The scenarios should be diverse, encompassing elements such as natural landscapes, historical settings, futuristic scenes, and imaginative contexts related to '{theme}'. Each element in the list should be a concise but descriptive scenario, designed to inspire visually rich images. Format these as elements in a Python list."
        }
    }

    if list_type == "text_questions":
        if len(instruct_questions) < num_questions_needed:
            for theme in template.INSTRUCT_DEFAULT_THEMES:
                for complexity_level in range(1, 11): 
                    for relevance_level in range(1, 11):
                        prompt = f"Generate a Python list of {prompts_in_question[list_type]} questions or instruct tasks related to the theme '{theme}', each with a complexity level of {complexity_level} out of 10 and a relevance level to the theme of {relevance_level} out of 10. These tasks should varyingly explore the theme in a manner that is consistent with their assigned complexity and relevance levels, allowing for a diverse and insightful engagement with the topic. Ensure that the output is formatted as elements in a Python list."
                        instruct_questions.append(prompt)

    selected_prompts = []
    for _ in range(math.ceil(num_questions_needed / prompts_in_question[list_type])):
        if list_type == "text_questions":
            prompt = random.choice(instruct_questions)
            instruct_questions.remove(prompt)
        elif list_type == "images_questions":
            prompt = list_type_mapping[list_type]["prompt"]

        selected_prompts.append(prompt)

    bt.logging.info(f"num_questions_needed is {num_questions_needed}")
    bt.logging.info(f"list_type is {list_type}")
    bt.logging.info(f"selected_prompts is {selected_prompts}")

    tasks = []
    for prompt in selected_prompts:
        random_seed = random.randint(1, 10000)
        messages = [{'role': "user", 'content': prompt}]
        task = call_openai(messages, 1, "gpt-4-1106-preview", random_seed)
        tasks.append(task)

    # Run all tasks concurrently and wait for them to complete
    responses = await asyncio.gather(*tasks)
    
    extracted_lists = []
    for answer in responses:
        answer = answer.replace("\n", " ") if answer else ""
        extracted_list = extract_python_list(answer)
        if extracted_list:
            bt.logging.success(f"Received new {list_type}")
            bt.logging.info(f"questions are {extracted_list}")
            extracted_lists += extracted_list
        else:
            bt.logging.info(f"No valid python list found in {answer}")
    
    return extracted_lists


async def call_openai(messages, temperature, model, seed=1234):
    for attempt in range(2):
        bt.logging.debug("Calling Openai ")
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                seed=seed,
            )
            response = response.choices[0].message.content
            bt.logging.debug(f"validator response is {response}")
            return response

        except Exception as e:
            bt.logging.error(f"Error when calling OpenAI: {e}")
            await asyncio.sleep(0.5) 
    
    return None


category = "text"
num_questions_needed = 30
themes = template.INSTRUCT_DEFAULT_THEMES
theme = random.choice(themes)


text1 = """
my_list = [     # Question 1     "Develop a comprehensive "algorithm" to make predictions, considering variables such as cast 'popularity' and marketing budget.",          # Question 2     "Create a neural network model that can generate ""original"" scripts for a TV series" ] 
"""

text2 = """
    ["1. Identify the syntax error in the following code: for i in range(10): print(i)", "2. Debug the code below to correctly calculate the factorial of a given number:\ndef factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)", "3. Fix the logical error in this code snippet that is intended to check if a number is prime:\ndef is_prime(n):\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True", "4. Analyze the code and identify the source of the IndexError:\ndef print_last_element(lst):\n    print(lst[-1])\n\nnumbers = [1, 2, 3]\nprint_last_element(numbers[3])", "5. Debug the code to correct the indentation error and ensure the print statement is executed:\ndef greet(name):\nprint(Hello, ", name)\ngreet(John)", "6. Identify and rectify the logical error in the code that is intended to find the sum of all even numbers in a list:\ndef sum_even_numbers(numbers):\ntotal = 0\nfor num in numbers:\nif num % 2 == 0:\ntotal += num\nreturn total", "7. Debug the code to resolve the TypeError:\ndef concatenate_strings(a, b):\nreturn a + b\n\nname = Alice\nage = 25\nresult = concatenate_strings(name, age)", "8. Analyze the code and identify and fix the NameError:\ndef calculate_average(numbers):\ntotal = sum(numbers)\naverage = total / count\nreturn average\n\ngrades = [90, 85, 95]\naverage_grade = calculate_average(grades)", "9. Debug the code to correctly reverse a given string:\ndef reverse_string(string):\nreversed_str = \nfor i in range(len(string)-1, -1, -1):\nreversed_str += string[i]\nreturn reversed_str\n\nprint(reverse_string(Python))", "10. Identify and fix the logical error in the code that is intended to sort a list of numbers in descending order:\ndef sort_descending(numbers):\nsorted_numbers = []\nfor num in numbers:\nif num > sorted_numbers[0]:\nsorted_numbers.insert(0, num)\nelse:\nsorted_numbers.append(num)\nreturn sorted_numbers"]
"""

print(extract_python_list(text2))

# async def main():
#     await get_list(f"{category}_questions", num_questions_needed, theme)

# if __name__ == "__main__":
#     asyncio.run(main())



