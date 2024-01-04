import asyncio
import concurrent
import itertools
import traceback
import random
from typing import Tuple, Union

import bittensor as bt
import torch
import wandb
from bittensor.btlogging import logger

from template.protocol import IsAlive
from validators import text_validator, image_validator
from validators.text_validator import TextValidator

iterations_per_set_weights = 12
scoring_organic_timeout = 60


async def wait_for_coro_with_limit(coro, timeout: int) -> Tuple[bool, object]:
    try:
        result = await asyncio.wait_for(coro, timeout)
    except asyncio.TimeoutError:
        logger.error('scoring task timed out')
        return False, None
    return True, result


class WeightSetter:
    def __init__(self, loop: asyncio.AbstractEventLoop, dendrite, subtensor, config, wallet, text_vali, image_vali):
        self.loop = loop
        self.dendrite = dendrite
        self.subtensor = subtensor
        self.config = config
        self.wallet = wallet
        self.text_vali = text_vali
        self.image_vali = image_vali

        self.moving_average_scores = None
        self.metagraph = subtensor.metagraph(config.netuid)
        self.total_scores = torch.zeros(len(self.metagraph.hotkeys))
        self.organic_scoring_tasks = set()

        self.thread_executor = concurrent.futures.ThreadPoolExecutor(thread_name_prefix='asyncio')
        self.loop.create_task(self.consume_organic_scoring())
        self.loop.create_task(self.perform_synthetic_scoring_and_update_weights())

    async def run_sync_in_async(self, fn):
        return await self.loop.run_in_executor(self.thread_executor, fn)

    async def consume_organic_scoring(self):
        while True:
            try:
                if self.organic_scoring_tasks:
                    completed, _ = await asyncio.wait(self.organic_scoring_tasks, timeout=1,
                                                      return_when=asyncio.FIRST_COMPLETED)
                    for task in completed:
                        if task.exception():
                            logger.error(
                                f'Encountered in {TextValidator.score_responses.__name__} task:\n'
                                f'{"".join(traceback.format_exception(task.exception()))}'
                            )
                        else:
                            success, data = task.result()
                            if not success:
                                continue
                            self.total_scores += data[0]
                    self.organic_scoring_tasks.difference_update(completed)
                else:
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f'Encountered in {self.consume_organic_scoring.__name__} loop:\n{traceback.format_exc()}')
                await asyncio.sleep(10)

    async def perform_synthetic_scoring_and_update_weights(self):
        while True:
            for steps_passed in itertools.count():
                self.metagraph = await self.run_sync_in_async(lambda: self.subtensor.metagraph(self.config.netuid))

                available_uids = await self.get_available_uids()
                selected_validator = self.select_validator(steps_passed)
                provider = self.select_provider(selected_validator, steps_passed)
                scores, _ = await self.process_modality(selected_validator, available_uids, provider)
                self.total_scores += scores

                steps_since_last_update = steps_passed % iterations_per_set_weights

                if steps_since_last_update == iterations_per_set_weights - 1:
                    await self.update_weights(steps_passed)
                else:
                    bt.logging.info(
                        f"Updating weights in {iterations_per_set_weights - steps_since_last_update - 1} iterations."
                    )

                await asyncio.sleep(0.5)

    def select_provider(self, validator, steps_passed) -> Union[text_validator.Provider, image_validator.Provider]:
        # TODO: implement
        return text_validator.Provider.openai

    def select_validator(self, steps_passed):
        return self.text_vali if steps_passed % 5 in (0, 1, 2) else self.image_vali

    async def get_available_uids(self):
        """Get a dictionary of available UIDs and their axons asynchronously."""
        tasks = {uid.item(): self.check_uid(self.metagraph.axons[uid.item()], uid.item()) for uid in self.metagraph.uids}
        results = await asyncio.gather(*tasks.values())

        # Create a dictionary of UID to axon info for active UIDs
        available_uids = {uid: axon_info for uid, axon_info in zip(tasks.keys(), results) if axon_info is not None}

        return available_uids

    async def check_uid(self, axon, uid):
        """Asynchronously check if a UID is available."""
        try:
            response = await self.dendrite(axon, IsAlive(), deserialize=False, timeout=4)
            if response.is_success:
                bt.logging.trace(f"UID {uid} is active")
                return axon  # Return the axon info instead of the UID

            bt.logging.trace(f"UID {uid} is not active")
            return None

        except Exception as e:
            bt.logging.error(f"Error checking UID {uid}: {e}\n{traceback.format_exc()}")
            return None

    def shuffled(self, list_: list) -> list:
        list_ = list_.copy()
        random.shuffle(list_)
        return list_

    async def process_modality(self, selected_validator, available_uids,
                               provider: Union[text_validator.Provider, image_validator.Provider]):
        uid_list = self.shuffled(list(available_uids.keys()))
        bt.logging.info(f"starting {selected_validator.__class__.__name__} get_and_score for {uid_list}")
        scores, uid_scores_dict, wandb_data = await selected_validator.get_and_score(uid_list, self.metagraph, provider)
        if self.config.wandb_on:
            wandb.log(wandb_data)
            bt.logging.success("wandb_log successful")
        return scores, uid_scores_dict

    async def update_weights(self, steps_passed):
        """ Update weights based on total scores, using min-max normalization for display. """
        avg_scores = self.total_scores / (steps_passed + 1)

        # Normalize avg_scores to a range of 0 to 1
        min_score = torch.min(avg_scores)
        max_score = torch.max(avg_scores)

        if max_score - min_score != 0:
            normalized_scores = (avg_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = torch.zeros_like(avg_scores)

        bt.logging.info(f"normalized_scores = {normalized_scores}")
        # We can't set weights with normalized scores because that disrupts the weighting assigned to each validator class
        # Weights get normalized anyways in weight_utils
        await self.set_weights(avg_scores)

    async def set_weights(self, scores):
        # alpha of .3 means that each new score replaces 30% of the weight of the previous weights
        alpha = .3
        if self.moving_average_scores is None:
            self.moving_average_scores = scores.clone()

        # Update the moving average scores
        self.moving_average_scores = alpha * scores + (1 - alpha) * self.moving_average_scores
        bt.logging.info(f"Updated moving average of weights: {self.moving_average_scores}")
        await self.run_sync_in_async(
            lambda: self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=self.metagraph.uids,
                weights=self.moving_average_scores,
                wait_for_inclusion=False,
            )
        )
        bt.logging.success("Successfully set weights.")

    def register_text_validator_organic_query(
        self,
        uid_to_response: dict[int, str],  # [(uid, response)]
        messages_dict: dict[int, str],
        provider: text_validator.Provider,
    ):
        self.organic_scoring_tasks.add(asyncio.create_task(
            wait_for_coro_with_limit(
                self.text_vali.score_responses(
                    query_responses=list(uid_to_response.items()),
                    uid_to_question=messages_dict,
                    metagraph=self.metagraph,
                    provider=provider,
                ),
                scoring_organic_timeout
            )
        ))


class TestWeightSetter(WeightSetter):
    def select_validator(self, steps_passed):
        return self.text_vali

    async def get_available_uids(self):
        return {i: None for i in range(len(self.metagraph.hotkeys))}

    def shuffled(self, list_: list) -> list:
        return list_
