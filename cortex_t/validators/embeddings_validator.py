from __future__ import annotations

import torch
import random
import asyncio
import bittensor as bt
import cortex_t.template.reward
from cortex_t.template import client
from datasets import load_dataset
from cortex_t.template.protocol import Embeddings
from cortex_t.validators.base_validator import BaseValidator


class EmbeddingsValidator(BaseValidator):
    def __init__(self, dendrite, config, subtensor, wallet):
        super().__init__(dendrite, config, subtensor, wallet, timeout=15)
        self.streaming = False
        self.query_type = "embeddings"
        self.model = "text-embedding-ada-002"
        self.weight = 1

        self.wandb_data = {
            "modality": "embeddings",
            "texts": {},
            "embeddings": {},
            "scores": {},
            "timestamps": {},
        }

    async def call_openai_embeddings(self, model, texts, batch_size=10):

        async def get_embeddings_in_batch(texts, model, batch_size=10):
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            tasks = []
            for batch in batches:
                filtered_batch = [text for text in batch if text.strip()]
                if filtered_batch:
                    # bt.logging.info("Log prompt.", filtered_batch)
                    task = asyncio.create_task(
                        client.embeddings.create(input=filtered_batch, model=model, encoding_format='float')
                    )
                    tasks.append(task)
                else:
                    bt.logging.info("Skipped an empty batch.")

            all_embeddings = []
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    bt.logging.error(f"Error in processing batch: {result}")
                else:
                    batch_embeddings = [item.embedding for item in result.data]
                    all_embeddings.extend(batch_embeddings)
            return all_embeddings

        all_embeddings = await get_embeddings_in_batch(texts, model)
        # for task in asyncio.as_completed(tasks):
        #     try:
        #         response = await task
        #         batch_embeddings = [item.embedding for item in response.data]
        #         all_embeddings.extend(batch_embeddings)
        #     except Exception as e:
        #         bt.logging.error(f"Error in processing batch: {e}")
        return all_embeddings

    def get_random_texts(self, dataset_name: str, config_name: str, num_samples: int = 100) -> list[str]:
        dataset = load_dataset(dataset_name, config_name)
        texts = [item['text'] for item in dataset['train']]
        return random.sample(texts, num_samples)

    async def start_query(self, available_uids, metagraph) -> tuple[list, dict]:
        if not available_uids:
            return [], {}

        query_tasks = []
        uid_to_question = {}
        random_texts = self.get_random_texts('wikitext', 'wikitext-2-v1', 100)
        num_texts_per_uid = len(random_texts) // len(available_uids)

        bt.logging.info(f"Each UID will receive {num_texts_per_uid} texts")

        for index, uid in enumerate(available_uids):
            start_index = index * num_texts_per_uid
            end_index = start_index + num_texts_per_uid
            prompt = random_texts[start_index:end_index]
            uid_to_question[uid] = prompt
            syn = Embeddings(model=self.model, texts=prompt)
            bt.logging.info(
                f"Sending {self.query_type} request to uid: {uid} "
                f"using {syn.model} with timeout {self.timeout}: {syn.texts[0]}"
            )
            task = self.query_miner(metagraph, uid, syn)
            query_tasks.append(task)
            self.wandb_data["texts"][uid] = prompt

        query_responses = await asyncio.gather(*query_tasks)
        return query_responses, uid_to_question

    async def score_responses(self, query_responses, uid_to_question, metagraph):
        scores = torch.zeros(len(metagraph.hotkeys))
        uid_scores_dict = {}
        embedding_score_tasks = []
        scoring_tasks = []

        random_number = random.random()
        will_score_all = random_number < 1/1.1
        bt.logging.info(f"Random Number: {random_number}, Will Score All: {will_score_all}")

        for uid, response in query_responses:
            if will_score_all and response:
                messages = uid_to_question[uid]
                task = self.call_openai_embeddings(self.model, messages)
                embedding_score_tasks.append((uid, task))

        # Await all embedding tasks
        embeddings_results = await asyncio.gather(*[task for _, task in embedding_score_tasks])

        # Now create new tasks for scoring embeddings
        for (uid, _), openai_answer in zip(embedding_score_tasks, embeddings_results):
            if not openai_answer:
                continue

            response = next(res for u, res in query_responses if u == uid)
            response = response[0]
            if response.embeddings is not None:
                task = cortex_t.template.reward.embeddings_score_dot(openai_answer, response.embeddings, self.weight)
                scoring_tasks.append((uid, task))
            else:
                scores[uid] = 0
                uid_scores_dict[uid] = 0

        # Await all scoring tasks
        scored_responses = await asyncio.gather(*[task for _, task in scoring_tasks])

        for (uid, _), score in zip(scoring_tasks, scored_responses):  # Use scoring_tasks here
            uid_scores_dict[uid] = scores[uid] = score or 0
            self.wandb_data["scores"][uid] = score

        return scores, uid_scores_dict, self.wandb_data
