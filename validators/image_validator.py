import io
import torch
import wandb
import random
import asyncio
import aiohttp
import base64
import traceback
import cortext.reward
import bittensor as bt

from PIL import Image
from io import BytesIO
from cortext.utils import get_question
from base_validator import BaseValidator
from cortext.protocol import ImageResponse


class ImageValidator(BaseValidator):
    def __init__(self, dendrite, config, subtensor, wallet):
        super().__init__(dendrite, config, subtensor, wallet, timeout=30)
        self.streaming = False
        self.query_type = "images"
        self.model = "dall-e-3"
        self.weight = .5
        self.provider = "OpenAI"
        self.size = "1792x1024"
        self.width = 1024
        self.height = 1024
        self.quality = "standard"
        self.style = "vivid"
        self.steps = 30
        self.seed = 123456
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
            providers = ["OpenAI"] * 100 + ["Stability"] * 0
            self.provider = random.choice(providers)

            if self.provider == "Stability":
                self.seed = random.randint(1000, 1000000)
                self.model = "stable-diffusion-xl-1024-v1-0"

            elif self.provider == "OpenAI":
                self.model = "dall-e-3"

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
        try:
            async with session.get(url) as response:
                content = await response.read()
                return await asyncio.to_thread(Image.open, BytesIO(content))
        except Exception as e:
            bt.logging.error(f"Exception occurred while downloading image: {traceback.format_exc()}")


    async def score_responses(self, _, query_responses, uid_to_question, metagraph):
        scores = torch.zeros(len(metagraph.hotkeys))
        uid_scores_dict = {}
        download_tasks = []
        score_tasks = []
        rand = random.random()
        will_score_all = rand < 1/1

        async with aiohttp.ClientSession() as session:
            try:
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
                            download_tasks.append(asyncio.create_task(self.download_image(image_url, session)))
                        else:  # Stability
                            b64s = completion["b64s"]
                            bt.logging.info(f"UID {uid} responded with an image")
                            for b64 in b64s:
                                download_tasks.append(asyncio.create_task(self.b64_to_image(b64)))

                        if will_score_all:
                            if syn.provider == "OpenAI":
                                score_task = cortext.reward.dalle_score(uid, image_url, self.size, syn.messages, self.weight)
                            else:
                                continue
                                score_task = cortext.reward.deterministic_score(uid, syn, self.weight)
                            score_tasks.append(asyncio.create_task(score_task))


                # Process download results
                try:
                    download_results = await asyncio.gather(*download_tasks)
                    for image, uid in zip(download_results, [uid for uid, _ in query_responses]):
                        self.wandb_data["images"][uid] = wandb.Image(image)
                except:
                    bt.logging.error(f"error in downloading images {traceback.exception_exc()}")

                # Process score results
                score_results = await asyncio.gather(*score_tasks, return_exceptions=True)
                for score, uid in zip(score_results, [uid for uid, _ in query_responses]):
                    try:
                        final_score = score if score is not None else 0
                        scores[uid] = uid_scores_dict[uid] = final_score
                    except Exception as e:
                        bt.logging.error(f"Error processing score for UID {uid}: {traceback.format_exc()}")

            except:
                bt.logging.debug(f"error in score_responses {traceback.format_exc()}")

        bt.logging.info(f"Final scores: {uid_scores_dict}")
        bt.logging.info("score_responses process completed.")
        return scores, uid_scores_dict, self.wandb_data
