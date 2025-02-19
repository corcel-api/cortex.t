import re
from loguru import logger
import subprocess
import json
import os
import urllib.request
import tempfile
from collections import deque
from asyncio import Lock
from PIL import Image
import torch
import numpy as np
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer

RECENT_URLS = deque(maxlen=10000)
RECENT_URLS_LOCK = Lock()

# URL patterns for both OpenAI and Azure DALL-E
OPENAI_URL_PATTERN = re.compile(
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

AZURE_URL_PATTERN = re.compile(
    r"^https://dalleprodsec\.blob\.core\.windows\.net/private/images/"
    r"[a-f0-9-]+/generated_\d+\.png\?"
    r"se=\d{4}-\d{2}-\d{2}T\d{2}%3A\d{2}%3A\d{2}Z&"
    r"sig=[A-Za-z0-9%/+=]+&"
    r"ske=\d{4}-\d{2}-\d{2}T\d{2}%3A\d{2}%3A\d{2}Z&"
    r"skoid=[a-f0-9-]+&"
    r"sks=b&"
    r"skt=\d{4}-\d{2}-\d{2}T\d{2}%3A\d{2}%3A\d{2}Z&"
    r"sktid=[a-f0-9-]+&"
    r"skv=\d{4}-\d{2}-\d{2}&"
    r"sp=r&"
    r"spr=https&"
    r"sr=b&"
    r"sv=\d{4}-\d{2}-\d{2}$"
)


class ClipSimilarity:
    model = AutoModel.from_pretrained("openai/clip-vit-base-patch16").to("cpu")
    processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")

    def __call__(self, image, prompt) -> float:
        image_emb = (
            self.model.get_text_features(
                **self.tokenizer([prompt], truncation=True, return_tensors="pt")
            )[0]
            .detach()
            .cpu()
            .numpy()
        )
        text_emb = (
            self.model.get_image_features(
                **self.processor([image], return_tensors="pt")
            )[0]
            .detach()
            .cpu()
            .numpy()
        )

        similarity = np.dot(image_emb, text_emb) / (
            np.linalg.norm(image_emb) * np.linalg.norm(text_emb)
        )
        return similarity


CLIP_SIMILARITY = ClipSimilarity()


def download_image(url, save_as):
    urllib.request.urlretrieve(url, save_as)


def load_exif_from_url(image_url: str) -> dict:
    """Load EXIF data from an image URL."""

    # Create temp file in /tmp directory
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name
            download_image(image_url, temp_path)
    except Exception as e:
        logger.error(f"Error downloading image: {e}")
        return {}, None

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
        image = Image.open(temp_path)
        os.remove(temp_path)
    except OSError as e:
        logger.warning(f"Error removing temp file: {e}")

    return metadata, image


async def dall_e_deterministic_score(image_url: str, prompt: str, size: str) -> float:
    """Score an image based on deterministic criteria and prompt similarity.

    Returns a score between 0 and 1 based on:
    - DALL-E URL pattern validation (both OpenAI and Azure)
    - Image uniqueness check
    - DALL-E metadata verification
    - Prompt similarity using CLIP
    """
    logger.info(f"Checking if {image_url} is in recent URLs")

    # Check if URL is already in recent URLs using async lock
    async with RECENT_URLS_LOCK:
        if image_url in RECENT_URLS:
            return 0

    # Validate URL pattern for both OpenAI and Azure
    is_openai = bool(OPENAI_URL_PATTERN.match(image_url))
    is_azure = bool(AZURE_URL_PATTERN.match(image_url))

    if not (is_openai or is_azure):
        logger.info("Image URL does not match either OpenAI or Azure DALL-E URL pattern")
        return 0

    # Add URL to recent URLs queue
    async with RECENT_URLS_LOCK:
        RECENT_URLS.append(image_url)

    # Check image metadata and calculate similarity
    exif_data, image = load_exif_from_url(image_url)
    
    # For OpenAI, we check the URL pattern only since metadata might vary
    # For Azure, we also rely on the URL pattern
    if not image:
        logger.info("Failed to load image")
        return 0

    try:
        image_size = image.size
        width, height = image_size
        logger.info(f"Image size: {image_size}, Requested size: {size}")
        if f"{width}x{height}" != size:
            logger.info("Image size does not match requested size")
            return 0
        
        logger.info("Calculating CLIP score")
        with torch.no_grad():
            score = CLIP_SIMILARITY(image, prompt)
            logger.info(f"Prompt: {prompt}")
            logger.info(f"CLIP score: {score}")
            if score > 0.225:
                return 1
            else:
                return 0

    except Exception as e:
        logger.error(f"Error calculating CLIP score: {e}")
        return 0


if __name__ == "__main__":
    url = "https://oaidalleapiprodscus.blob.core.windows.net/private/org-D1M4iKyWTgllT9IRx1f0IZ1o/user-eKhw8tXOQNWX5bhjzwGqKJRb/img-YzvOGexdbWo6EvbDOCUMrPxL.png?st=2025-01-24T07%3A18%3A09Z&se=2025-01-24T09%3A18%3A09Z&sp=r&sv=2024-08-04&sr=b&rscd=inline&rsct=image/png&skoid=d505667d-d6c1-4a0a-bac7-5c84a87759f8&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-01-24T00%3A22%3A14Z&ske=2025-01-25T00%3A22%3A14Z&sks=b&skv=2024-08-04&sig=0IwWSzv2YitT7P6%2BgfosJzlZzzwTFxOqwNHbMWk5xeo%3D"
    # url = "https://oaidalleapiprodscus.blob.core.windows.net/private/org-D1M4iKyWTgllT9IRx1f0IZ1o/user-eKhw8tXOQNWX5bhjzwGqKJRb/img-jdeefxfcK4yPlnxmcjmAFsyw.png?st=2025-01-24T07%3A26%3A45Z&se=2025-01-24T09%3A26%3A45Z&sp=r&sv=2024-08-04&sr=b&rscd=inline&rsct=image/png&skoid=d505667d-d6c1-4a0a-bac7-5c84a87759f8&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-01-24T00%3A39%3A57Z&ske=2025-01-25T00%3A39%3A57Z&sks=b&skv=2024-08-04&sig=O3Hb7wd4FWsSFBPiQ9u5k6BzYvk/0AMlB04xtDwWNS4%3D"

    score = dall_e_deterministic_score(url, "a cat", "1024x1024")
    print(score)
