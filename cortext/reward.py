# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
from __future__ import annotations
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()


import re
import io
import torch
import openai
import typing
import difflib
import asyncio
import logging
import aiohttp
import requests
import traceback
import numpy as np
from numpy.linalg import norm
import bittensor as bt
from cortext import utils
from PIL import Image
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import CLIPProcessor, CLIPModel

# ==== TEXT ====

def calculate_text_similarity(text1: str, text2: str):
    try:
        # Initialize the TF-IDF Vectorizer
        vectorizer = TfidfVectorizer()

        # Vectorize the texts
        tfidf_matrix = vectorizer.fit_transform([text1, text2])

        # Calculate the Cosine Similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        bt.logging.debug(f"Similarity: {similarity}")
        return similarity
    except Exception as e:
        bt.logging.error(f"Error in calculate_text_similarity: {traceback.format_exc()}")
        raise

async def api_score(api_answer: str, response: str, weight: float, temperature: float, provider: str) -> float:
    try:
        loop = asyncio.get_running_loop()
        similarity = await loop.run_in_executor(None, calculate_text_similarity, api_answer, response)

        words_in_response = len(response.split())
        words_in_api = len(api_answer.split())

        word_count_over_threshold = words_in_api * 1.20
        word_count_under_threshold = words_in_api * 0.60

        # Check if the word count of the response is within the thresholds
        if words_in_response <= word_count_over_threshold and words_in_response >= word_count_under_threshold:
            score = weight * similarity
        else:
            score = 0

        return score
    except Exception as e:
        bt.logging.error(f"Exception in api_score: {traceback.format_exc()}")


# ==== IMAGES =====

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Could also verify the date from the url
url_regex = (
    r'https://(?:oaidalleapiprodscus|dalleprodsec)\.blob\.core\.windows\.net/private/org-[\w-]+/'
    r'user-[\w-]+/img-[\w-]+\.(?:png|jpg)\?'
    r'st=\d{4}-\d{2}-\d{2}T\d{2}%3A\d{2}%3A\d{2}Z&'
    r'se=\d{4}-\d{2}-\d{2}T\d{2}%3A\d{2}%3A\d{2}Z&'
    r'(?:sp=\w+&)?'
    r'sv=\d{4}-\d{2}-\d{2}&'
    r'sr=\w+&'
    r'rscd=\w+&'
    r'rsct=\w+/[\w-]+&'
    r'skoid=[\w-]+&'
    r'sktid=[\w-]+&'
    r'skt=\d{4}-\d{2}-\d{2}T\d{2}%3A\d{2}%3A\d{2}Z&'
    r'ske=\d{4}-\d{2}-\d{2}T\d{2}%3A\d{2}%3A\d{2}Z&'
    r'sks=\w+&'
    r'skv=\d{4}-\d{2}-\d{2}&'
    r'sig=[\w/%+=]+'
)


async def is_image_url(url: str) -> bool:
    """Check if the URL points to an image asynchronously."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(url) as response:
                return response.status == 200 and 'image' in response.headers.get('Content-Type', '')
    except Exception as e:
        bt.logging.info(f"Error checking URL: {e}")
        return False


async def load_image_from_url(url: str):
    """Load an image from a URL asynchronously."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                image_data = await response.read()
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                image.verify()  # Verify that this is indeed an image
                return image
    except Exception as e:
        bt.logging.info(f"Failed to load image: {e}")


def get_image_size(image) -> tuple[int, int]:
    """Get the size of an image."""
    return image.size  # Returns a tuple (width, height)


def calculate_image_similarity(image, description, max_length: int = 77):
    """Calculate the cosine similarity between a description and an image."""
    # Truncate the description
    inputs = processor(
        text=description, images=None, return_tensors="pt",
        padding=True, truncation=True, max_length=max_length,
    )
    text_embedding = model.get_text_features(**inputs)

    # Process the image
    inputs = processor(text=None, images=image, return_tensors="pt", padding=True, truncation=True)
    image_embedding = model.get_image_features(**inputs)

    # Calculate cosine similarity
    return torch.cosine_similarity(image_embedding, text_embedding, dim=1).item()

async def dalle_score(uid, url, desired_size, description, weight, similarity_threshold=0.23) -> float:
    """Calculate the image score based on similarity and size asynchronously."""

    if not re.match(url_regex, url):
        bt.logging.info(f"UID {uid} URL does not match the expected format.")
        return 0

    if not await is_image_url(url):
        bt.logging.info(f"UID {uid} URL does not point to a valid image.")
        return 0

    image = await load_image_from_url(url)
    if image is None:
        bt.logging.info(f"UID {uid} failed to load image from URL.")
        return 0

    size = get_image_size(image)
    size_str = f"{size[0]}x{size[1]}"
    if desired_size != size_str:
        bt.logging.info(f"UID {uid} size does not match: {size_str} != {desired_size} ")

    try:
        similarity = await asyncio.to_thread(calculate_image_similarity, image, description)
        if similarity > similarity_threshold:
            bt.logging.debug(
                f"UID {uid} passed similarity test with score of: {round(similarity, 5)}. "
                f"Score = {weight}"
            )
            return weight

        bt.logging.debug(f"UID {uid} failed similary test with score of: {round(similarity, 5)}. Score = {0}")
        return 0
    except Exception as e:
        bt.logging.info(f"Error in image scoring for UID {uid}: {e}")
        return 0



# IMAGES ---- DETERMINISTIC

async def deterministic_score(uid: int, syn, weight: float):
    vali_b64s = await utils.call_stability(syn.messages, syn.seed, syn.steps, syn.cfg_scale, syn.width, syn.height, syn.samples, syn.sampler)

    for miner_b64, vali_b64 in zip(syn.completion["b64s"], vali_b64s):
        if miner_b64[:50] != vali_b64[:50]:
            bt.logging.info(f"image for UID {uid} does not match the correct image! Score = 0")
            return 0

    bt.logging.info(f"returned image for UID {uid} matches the correct image! Score = {weight}")
    return weight



# ==== Embeddings =====

async def embeddings_score(openai_answer: list, response: list, weight: float, threshold: float = .95) -> float:
    if len(openai_answer) != len(response):
        bt.logging.info("The number of embeddings in openai_answer and response do not match.")
        return 0


    # Calculate similarity for each pair of embeddings
    similarities = []
    for oa_emb, resp_emb in zip(openai_answer, response):
        similarity = 1 - cosine(oa_emb, resp_emb)
        similarities.append(similarity)

    # Average the similarities
    avg_similarity = sum(similarities) / len(similarities)
    bt.logging.info(f"Average similarity: {avg_similarity}")

    # Check against thresholdls

    if avg_similarity > threshold:
        bt.logging.info("Average embeddings similarity exceeds threshold!")
        return weight

    bt.logging.info(f"Average embeddings similarity does not exceed threshold: {avg_similarity}")
    return 0


async def embeddings_score_dot(openai_answer: list, response: list, weight: float, threshold: float = .95) -> float:
    if len(openai_answer) != len(response):
        bt.logging.warning("The number of embeddings in openai_answer and response do not match.")
        return 0

    # Calculate cosine similarity for each pair of embeddings
    cosine_similarities = []
    for oa_emb, resp_emb in zip(openai_answer, response):
        if norm(oa_emb) == 0 or norm(resp_emb) == 0:
            bt.logging.error("One of the embeddings is a zero vector.")
            return 0
        cosine_similarity = np.dot(oa_emb, resp_emb) / (norm(oa_emb) * norm(resp_emb))
        cosine_similarity = min(1.0, max(cosine_similarity, -1.0))  # Clamp the value to the range [-1, 1]

        cosine_similarities.append(cosine_similarity)

    # Average the cosine similarities
    avg_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities)
    bt.logging.info(f"Average similarity: {avg_cosine_similarity}")

    # Check against threshold
    if avg_cosine_similarity > threshold:
        bt.logging.info("Average embeddings cosine similarity exceeds threshold!")
        return weight

    bt.logging.info(f"Average embeddings cosine similarity does not exceed threshold: {avg_cosine_similarity}")
    return 0
