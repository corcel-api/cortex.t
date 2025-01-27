import re
import requests
from loguru import logger
import subprocess
import json
import os
import urllib.request
import tempfile
from openai import AsyncOpenAI

VISION_CLIENT = AsyncOpenAI(
    api_key=os.getenv("TOGETHER_API_KEY"), base_url="https://api.together.xyz/v1"
)


def download_image(url, save_as):
    urllib.request.urlretrieve(url, save_as)


def load_exif_from_url(image_url: str) -> dict:
    """Load EXIF data from an image URL."""

    # Create temp file in /tmp directory
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_path = temp_file.name
        download_image(image_url, temp_path)

    # Run exiftool and capture output
    try:
        result = subprocess.run(
            ["exiftool", "-json", temp_path], capture_output=True, text=True, check=True
        )
        metadata = json.loads(result.stdout)[0]
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running exiftool: {e}")
        metadata = {}
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing exiftool output: {e}")
        metadata = {}

    try:
        os.remove(temp_path)
    except OSError as e:
        logger.warning(f"Error removing temp file: {e}")

    return metadata


async def dall_e_deterministic_score(image_url: str, prompt: str, size: str) -> int:
    """Score an image based on its deterministic score.

    Validates if the URL matches the expected DALL-E API URL pattern.
    Returns 1 if valid, 0 if invalid.
    """
    dalle_url_pattern = re.compile(
        r"^https://oaidalleapiprodscus\.blob\.core\.windows\.net/private/"
        r"org-[A-Za-z0-9]+/user-[A-Za-z0-9]+/img-[A-Za-z0-9]+\.png\?"
        r"st=\d{4}-\d{2}-\d{2}T\d{2}%3A\d{2}%3A\d{2}Z&"
        r"se=\d{4}-\d{2}-\d{2}T\d{2}%3A\d{2}%3A\d{2}Z&"
        r"sp=r&sv=\d{4}-\d{2}-\d{2}&sr=b&rscd=inline&rsct=image/png&"
        r"skoid=[a-f0-9-]+&sktid=[a-f0-9-]+&"
        r"skt=\d{4}-\d{2}-\d{2}T\d{2}%3A\d{2}%3A\d{2}Z&"
        r"ske=\d{4}-\d{2}-\d{2}T\d{2}%3A\d{2}%3A\d{2}Z&"
        r"sks=b&skv=\d{4}-\d{2}-\d{2}&"
        r"sig=[A-Za-z0-9%/+=]+$"
    )

    if not dalle_url_pattern.match(image_url):
        return 0

    exif_data = load_exif_from_url(image_url)

    # Differentiate between DALL-E 2 and DALL-E 3
    if not "Claim_generator" in exif_data:
        return 0

    scoring_prompt = f"""
You are an AI tasked with evaluating the adherence of generated images to their corresponding prompt strings. Your job is to analyze how well the image matches the given prompt and assign a score from 0 to 10, where 0 indicates no adherence and 10 indicates perfect adherence.

Please carefully examine the image and compare it to the prompt string. Consider the following aspects:
1. How many elements from the prompt are present in the image?
2. How accurately are these elements depicted?
3. Does the overall composition and mood of the image match the prompt?
4. Are there any significant elements in the image that were not mentioned in the prompt?

Based on your analysis, determine a score from 0 to 10 that represents how well the image adheres to the prompt. Use this scale as a guide:
0-2: Poor adherence, major discrepancies
3-4: Below average adherence, significant mismatches
5-6: Average adherence, some elements match but others are missing or inaccurate
7-8: Good adherence, most elements match with minor discrepancies
9-10: Excellent adherence, nearly perfect or perfect match

Provide only the numerical score (0-10) inside <score> tags. Do not include any additional text with the score.

Here is the prompt string used to generate the image:
<prompt_string>
{prompt}
</prompt_string>
"""
    scoring_prompt = scoring_prompt.replace("{{PROMPT_STRING}}", prompt)
    output = await VISION_CLIENT.chat.completions.create(
        model="Qwen/Qwen2-VL-72B-Instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": scoring_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            }
        ],
        stream=False,
    )
    logger.info(output)
    completion = output.choices[0].message.content
    logger.info(completion)
    score = re.search(r"<score>(.*?)</score>", completion).group(1)
    return float(score) / 10


if __name__ == "__main__":
    url = "https://oaidalleapiprodscus.blob.core.windows.net/private/org-D1M4iKyWTgllT9IRx1f0IZ1o/user-eKhw8tXOQNWX5bhjzwGqKJRb/img-YzvOGexdbWo6EvbDOCUMrPxL.png?st=2025-01-24T07%3A18%3A09Z&se=2025-01-24T09%3A18%3A09Z&sp=r&sv=2024-08-04&sr=b&rscd=inline&rsct=image/png&skoid=d505667d-d6c1-4a0a-bac7-5c84a87759f8&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-01-24T00%3A22%3A14Z&ske=2025-01-25T00%3A22%3A14Z&sks=b&skv=2024-08-04&sig=0IwWSzv2YitT7P6%2BgfosJzlZzzwTFxOqwNHbMWk5xeo%3D"
    # url = "https://oaidalleapiprodscus.blob.core.windows.net/private/org-D1M4iKyWTgllT9IRx1f0IZ1o/user-eKhw8tXOQNWX5bhjzwGqKJRb/img-jdeefxfcK4yPlnxmcjmAFsyw.png?st=2025-01-24T07%3A26%3A45Z&se=2025-01-24T09%3A26%3A45Z&sp=r&sv=2024-08-04&sr=b&rscd=inline&rsct=image/png&skoid=d505667d-d6c1-4a0a-bac7-5c84a87759f8&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-01-24T00%3A39%3A57Z&ske=2025-01-25T00%3A39%3A57Z&sks=b&skv=2024-08-04&sig=O3Hb7wd4FWsSFBPiQ9u5k6BzYvk/0AMlB04xtDwWNS4%3D"

    score = image_deterministic_score(url)
    print(score)
