import re
from loguru import logger
import subprocess
import json
import os
import urllib.request
import tempfile
from openai import AsyncOpenAI

VISION_CLIENT = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1"
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
    if "Claim_generator" not in exif_data:
        return 0

    scoring_prompt = f"""
    Your are provided with an image and a alt text.
    Your task is to determine if the image is related to the alt text.
    Please return "yes" if the image is related to the alt text, otherwise return "no".
    Don't explain anything, just return "yes" or "no".
    ---
    Prompt: "{prompt}"
    """
    scoring_prompt = scoring_prompt.replace("{{PROMPT_STRING}}", prompt)
    output = await VISION_CLIENT.chat.completions.create(
        model="qwen/qwen-2-vl-72b-instruct",
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
    words = completion.lower().split()
    words = [re.sub(r"[^a-zA-Z]", "", word) for word in words]
    logger.info(words)
    score = "yes" in words and "no" not in words
    return float(score)


if __name__ == "__main__":
    url = "https://oaidalleapiprodscus.blob.core.windows.net/private/org-D1M4iKyWTgllT9IRx1f0IZ1o/user-eKhw8tXOQNWX5bhjzwGqKJRb/img-YzvOGexdbWo6EvbDOCUMrPxL.png?st=2025-01-24T07%3A18%3A09Z&se=2025-01-24T09%3A18%3A09Z&sp=r&sv=2024-08-04&sr=b&rscd=inline&rsct=image/png&skoid=d505667d-d6c1-4a0a-bac7-5c84a87759f8&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-01-24T00%3A22%3A14Z&ske=2025-01-25T00%3A22%3A14Z&sks=b&skv=2024-08-04&sig=0IwWSzv2YitT7P6%2BgfosJzlZzzwTFxOqwNHbMWk5xeo%3D"
    # url = "https://oaidalleapiprodscus.blob.core.windows.net/private/org-D1M4iKyWTgllT9IRx1f0IZ1o/user-eKhw8tXOQNWX5bhjzwGqKJRb/img-jdeefxfcK4yPlnxmcjmAFsyw.png?st=2025-01-24T07%3A26%3A45Z&se=2025-01-24T09%3A26%3A45Z&sp=r&sv=2024-08-04&sr=b&rscd=inline&rsct=image/png&skoid=d505667d-d6c1-4a0a-bac7-5c84a87759f8&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-01-24T00%3A39%3A57Z&ske=2025-01-25T00%3A39%3A57Z&sks=b&skv=2024-08-04&sig=O3Hb7wd4FWsSFBPiQ9u5k6BzYvk/0AMlB04xtDwWNS4%3D"

    score = dall_e_deterministic_score(url, "a cat", "1024x1024")
    print(score)
