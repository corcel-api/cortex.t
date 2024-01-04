from abc import ABC, abstractmethod

import bittensor as bt


class BaseValidator(ABC):
    def __init__(self, dendrite, config, subtensor, wallet, timeout):
        self.dendrite = dendrite
        self.config = config
        self.subtensor = subtensor
        self.wallet = wallet
        self.timeout = timeout
        self.streaming = False

    async def query_miner(self, metagraph, uid, syn):
        try:
            responses = await self.dendrite([metagraph.axons[uid]], syn, deserialize=False, timeout=self.timeout,
                                            streaming=self.streaming)
            return await self.handle_response(uid, responses)

        except Exception as e:
            bt.logging.error(f"Exception during query for uid {uid}: {e}")
            return uid, None

    async def handle_response(self, uid, responses):
        return uid, responses

    @abstractmethod
    async def start_query(self, available_uids) -> tuple[list, dict]:
        ...

    @abstractmethod
    async def score_responses(self, responses):
        ...

    async def get_and_score(self, available_uids, metagraph, provider):
        query_responses, uid_to_question = await self.start_query(available_uids, metagraph, provider)
        return await self.score_responses(query_responses, uid_to_question, metagraph, provider)
