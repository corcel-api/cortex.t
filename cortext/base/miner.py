import bittensor as bt
from abc import ABC, abstractmethod
import threading
from loguru import logger
import argparse
import os
from .config import add_common_config
from typing import Callable


class BaseMiner(ABC):
    def __init__(self, attach_fns: list[tuple[Callable, Callable]] = []):
        self.config = self.get_config()
        self.attach_fns = [
            (self.forward, self.blacklist),
        ] + attach_fns
        self.init_bittensor()

    def get_config(self):
        parser = argparse.ArgumentParser()
        parser = add_common_config(parser)
        parser.add_argument("--miner.total_credit", type=int, default=128)
        config = bt.config(parser)
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey_str,
                config.netuid,
                "validator",
            )
        )
        print(config)
        os.makedirs(config.full_path, exist_ok=True)
        return config

    def init_bittensor(self):
        self.subtensor = bt.subtensor(config=self.config)
        self.wallet = bt.wallet(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        self.axon = bt.axon(config=self.config)
        for forward_fn, blacklist_fn in self.attach_fns:
            self.axon.attach(forward_fn=forward_fn, blacklist_fn=blacklist_fn)
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()

    def chain_sync(self):
        self.metagraph.sync()

    @abstractmethod
    async def forward(self, synapse: bt.Synapse) -> bt.Synapse: ...

    @abstractmethod
    async def blacklist(self, synapse: bt.Synapse) -> bool: ...
