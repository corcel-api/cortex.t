import asyncio
import random
from typing import AsyncIterator

import bittensor as bt
import torch
from cortex_t.validators.base_validator import BaseValidator

import cortex_t.template.reward
from cortex_t.template.protocol import StreamPrompting
from cortex_t.template.utils import call_openai, get_question, call_anthropic


class TextValidator(BaseValidator):
    def __init__(self, dendrite, config, subtensor, wallet: bt.wallet):
        super().__init__(dendrite, config, subtensor, wallet, timeout=75)
        self.streaming = True
        self.query_type = "text"
        self.model =  "gpt-3.5-turbo" # "gpt-4-1106-preview"
        self.max_tokens = 2048
        self.temperature = 0.0001
        self.weight = 1
        self.seed = 1234
        self.top_p = 0.01
        self.top_k = 1
        self.provider = "OpenAI"

        self.wandb_data = {
            "modality": "text",
            "prompts": {},
            "responses": {},
            "scores": {},
            "timestamps": {},
        }

    async def organic(self, metagraph, query: dict[str, list[dict[str, str]]]) -> AsyncIterator[tuple[int, str]]:
        for uid, messages in query.items():
            syn = StreamPrompting(messages=messages, model=self.model, seed=self.seed, max_tokens=self.max_tokens, temperature=self.temperature, provider=self.provider, top_p=self.top_p, top_k=self.top_k)
            bt.logging.info(
                f"Sending {syn.model} {self.query_type} request to uid: {uid}, "
                f"timeout {self.timeout}: {syn.messages[0]['content']}"
            )

            self.wandb_data["prompts"][uid] = messages
            responses = await self.dendrite(
                metagraph.axons[uid],
                syn,
                deserialize=False,
                timeout=self.timeout,
                streaming=self.streaming,
            )

            async for resp in responses:
                if not isinstance(resp, str):
                    continue

                bt.logging.trace(resp)
                yield uid, resp

    async def handle_response(self, uid: str, responses) -> tuple[str, str]:
        full_response = ""
        for resp in responses:
            async for chunk in resp:
                if isinstance(chunk, str):
                    bt.logging.trace(chunk)
                    full_response += chunk
            bt.logging.debug(f"full_response for uid {uid}: {full_response}")
            break
        return uid, full_response

    async def get_question(self, qty):
        return await get_question("text", qty)

    def get_random_provider(self):
        return random.choice(["OpenAI", "Anthropic"])

    async def start_query(self, available_uids, metagraph) -> tuple[list, dict]:
        query_tasks = []
        uid_to_question = {}
        # Randomly choose the provider based on specified probabilities
        self.provider = self.get_random_provider()

        if self.provider == "Anthropic":
            # bedrock models = ["anthropic.claude-v2:1", "anthropic.claude-instant-v1", "anthropic.claude-v1", "anthropic.claude-v2"]
            # claude models = ["claude-2.1", "claude-2.0", "claude-instant-1.2"]
            self.model = "anthropic.claude-instant-v1"
        elif self.provider == "OpenAI":
            self.model = "gpt-3.5-turbo"

        for uid in available_uids:
            prompt = await self.get_question(len(available_uids))
            uid_to_question[uid] = prompt
            messages = [{'role': 'user', 'content': prompt}]
            syn = StreamPrompting(messages=messages, model=self.model, seed=self.seed, max_tokens=self.max_tokens, temperature=self.temperature, provider=self.provider, top_p=self.top_p, top_k=self.top_k)
            bt.logging.info(
                f"Sending {syn.model} {self.query_type} request to uid: {uid}, "
                f"timeout {self.timeout}: {syn.messages[0]['content']}"
            )
            task = self.query_miner(metagraph, uid, syn)
            query_tasks.append(task)
            self.wandb_data["prompts"][uid] = prompt

        query_responses = await asyncio.gather(*query_tasks)
        return query_responses, uid_to_question

    def should_i_score(self):
        random_number = random.random()
        will_score_all = random_number < 1 / 2
        bt.logging.info(f"Random Number: {random_number}, Will score text responses: {will_score_all}")
        return will_score_all

    async def call_openai(self, prompt: str) -> str:
        return await call_openai([{'role': 'user', 'content': prompt}], 0, self.temperature, self.model,
                                 self.seed, self.max_tokens)

    async def call_api(self, prompt: str, provider: str) -> str:
        if provider == "OpenAI":
            return await self.call_openai(prompt)
        elif provider == "Anthropic":
            return await call_anthropic(prompt, self.temperature, self.model, self.max_tokens, self.top_p, self.top_k)
        else:
            bt.logging.error(f"provider {provider} not found")

    async def score_responses(
        self,
        query_responses: list[tuple[int, str]],  # [(uid, response)]
        uid_to_question: dict[int, str],  # uid -> prompt
        metagraph: bt.metagraph,
    ) -> tuple[torch.Tensor, dict[int, float], dict]:
        scores = torch.zeros(len(metagraph.hotkeys))
        uid_scores_dict = {}
        response_tasks = []

        # Decide to score all UIDs this round based on a chance
        will_score_all = self.should_i_score()

        for uid, response in query_responses:
            self.wandb_data["responses"][uid] = response
            if will_score_all and response:
                prompt = uid_to_question[uid]
                response_tasks.append((uid, self.call_api(prompt, self.provider)))

        api_responses = await asyncio.gather(*[task for _, task in response_tasks])

        scoring_tasks = []
        for (uid, _), api_answer in zip(response_tasks, api_responses):
            if api_answer:
                response = next(res for u, res in query_responses if u == uid)  # Find the matching response
                task = cortex_t.template.reward.api_score(api_answer, response, self.weight)
                scoring_tasks.append((uid, task))

        scored_responses = await asyncio.gather(*[task for _, task in scoring_tasks])

        bt.logging.debug(f"scored responses = {scored_responses}")
        for (uid, _), scored_response in zip(scoring_tasks, scored_responses):
            if scored_response is not None:
                scores[uid] = scored_response
                uid_scores_dict[uid] = scored_response
            else:
                scores[uid] = 0
                uid_scores_dict[uid] = 0
            # self.wandb_data["scores"][uid] = score

        if uid_scores_dict != {}:
            bt.logging.info(f"text_scores is {uid_scores_dict}")
        return scores, uid_scores_dict, self.wandb_data


class TestTextValidator(TextValidator):
    def __init__(
        self,
        dendrite,
        config,
        subtensor,
        wallet: bt.wallet,
    ):
        super().__init__(dendrite, config, subtensor, wallet)
        self.openai_prompt_to_contents: dict[str, list[str]] = {}
        self.questions: list[str] = []
        self._questions_retrieved = -1
        self._openai_prompts_used: dict[str, int] = {}

    def feed_mock_data(self, openai_prompt_to_contents, questions):
        self.questions = questions
        self.openai_prompt_to_contents = openai_prompt_to_contents
        self._openai_prompts_used = dict.fromkeys(self.openai_prompt_to_contents, -1)
        self._questions_retrieved = -1

    def get_random_provider(self):
        return "OpenAI"

    def should_i_score(self):
        return True

    async def call_openai(self, prompt: str) -> str:
        self._openai_prompts_used[prompt] += 1
        used = self._openai_prompts_used[prompt]
        contents = self.openai_prompt_to_contents[prompt]
        return contents[used % len(contents)]

    async def get_question(self, qty):
        self._questions_retrieved += 1
        return self.questions[self._questions_retrieved % len(self.questions)]

    async def query_miner(self, metagraph, uid, syn: StreamPrompting):
        return uid, await self.call_openai(syn.messages[0]['content'])

    async def organic(self, metagraph, query: dict[str, list[dict[str, str]]]) -> AsyncIterator[tuple[int, str]]:
        for uid, messages in query.items():
            for msg in messages:
                yield uid, await self.call_openai(msg['content'])
