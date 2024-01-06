import torch
import wandb
import random
import asyncio
import aiohttp
import base64
import traceback
import cortex_t.template.reward
import bittensor as bt

from PIL import Image
from io import BytesIO
from cortex_t.template.utils import get_question
from cortex_t.validators.base_validator import BaseValidator
from cortex_t.template.protocol import ImageResponse


class ImageValidator(BaseValidator):
    def __init__(self, dendrite, config, subtensor, wallet):
        super().__init__(dendrite, config, subtensor, wallet, timeout=60)
        self.streaming = False
        self.query_type = "images"
        self.model = "dall-e-2"
        self.weight = .5
        self.provider = "OpenAI"
        self.size = "1024x1024"
        self.width = 1024
        self.height = 1024
        self.quality = "standard"
        self.style = "vivid"
        self.steps = 30
        self.seed = 12345
        self.wandb_data = {
            "modality": "images",
            "prompts": {},
            "responses": {},
            "images": {},
            "scores": {},
            "timestamps": {},
        }

    async def start_query(self, available_uids, metagraph):
        try:
            query_tasks = []
            uid_to_question = {}

            # Randomly choose the provider based on specified probabilities
            providers = ["OpenAI"] * 5 + ["Stability"] * 5
            self.provider = random.choice(providers)

            if self.provider == "Stability":
                self.seed = random.randint(1000, 1000000)
                self.model = "stable-diffusion-xl-1024-v1-0"

            elif self.provider == "OpenAI":
                self.model = "dall-e-2"

            # Query all images concurrently
            for uid in available_uids:
                messages = await get_question("images", len(available_uids))
                uid_to_question[uid] = messages  # Store messages for each UID

                syn = ImageResponse(messages=messages, model=self.model, size=self.size, quality=self.quality, style=self.style, provider=self.provider, seed=self.seed, steps=self.steps)
                bt.logging.info(f"uid = {uid}, syn = {syn}")

                # bt.logging.info(
                #     f"Sending a {self.size} {self.quality} {self.style} {self.query_type} request "
                #     f"to uid: {uid} using {syn.model} with timeout {self.timeout}: {syn.messages}"
                # )
                task = self.query_miner(metagraph, uid, syn)
                query_tasks.append(task)
                self.wandb_data["prompts"][uid] = messages

            # Query responses is (uid. syn)
            query_responses = await asyncio.gather(*query_tasks)
            return query_responses, uid_to_question
        except:
            bt.logging.error(f"error in start_query {traceback.format_exc()}")


    async def b64_to_image(self, b64):
        image_data = base64.b64decode(b64)
        return await asyncio.to_thread(Image.open, BytesIO(image_data))

    async def download_image(self, url, session):
        async with session.get(url) as response:
            content = await response.read()
            return await asyncio.to_thread(Image.open, BytesIO(content))

    async def process_download_result(self, uid, download_task):
        try:
            image = await download_task
            self.wandb_data["images"][uid] = wandb.Image(image)
        except Exception as e:
            bt.logging.error(f"Error downloading image for UID {uid}: {e}")

    async def process_score_result(self, uid, score_task, scores, uid_scores_dict):
        try:
            scored_response = await score_task
            score = scored_response if scored_response is not None else 0
            scores[uid] = uid_scores_dict[uid] = score
        except Exception as e:
            bt.logging.error(f"Error scoring image for UID {uid}: {e}")

    async def score_responses(self, query_responses, uid_to_question, metagraph):
        scores = torch.zeros(len(metagraph.hotkeys))
        uid_scores_dict = {}
        download_tasks = []
        score_tasks = []
        rand = random.random()
        will_score_all = rand < 1/1
        bt.logging.info(f"random number = {rand}, will score all = {will_score_all}")
        async with aiohttp.ClientSession() as session:
            for uid, syn in query_responses:
                syn = syn[0]
                completion = syn.completion
                if completion is None:
                    scores[uid] = uid_scores_dict[uid] = 0
                    continue

                if syn.provider in ["OpenAI", "Stability"]:
                    if syn.provider == "OpenAI":
                        image_url = completion["url"]
                        bt.logging.info(f"UID {uid} response = {image_url}")
                        download_tasks.append((uid, asyncio.create_task(self.download_image(image_url, session))))
                    else:  # Stability
                        b64s = completion["b64s"]
                        bt.logging.info(f"UID {uid} responded with an image")
                        for b64 in b64s:
                            download_tasks.append((uid, asyncio.create_task(self.b64_to_image(b64))))

                    if will_score_all:
                        if syn.provider == "OpenAI":
                            score_task = cortex_t.template.reward.dalle_score(uid, image_url, self.size, syn.messages, self.weight)
                        else:
                            score_task = cortex_t.template.reward.deterministic_score(uid, syn, self.weight)

                        score_tasks.append((uid, asyncio.create_task(score_task)))

            await asyncio.gather(*(dt[1] for dt in download_tasks), *(st[1] for st in score_tasks))

        download_results = [self.process_download_result(uid, dt) for uid, dt in download_tasks]
        await asyncio.gather(*download_results)

        score_results = [self.process_score_result(uid, st, scores, uid_scores_dict) for uid, st in score_tasks]
        await asyncio.gather(*score_results)

        if uid_scores_dict != {}:
            bt.logging.info(f"scores = {uid_scores_dict}")

        return scores, uid_scores_dict, self.wandb_data

