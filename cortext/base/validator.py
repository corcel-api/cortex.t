import os
import asyncio
import argparse
import traceback
import bittensor as bt
import time
import threading
from abc import abstractmethod, ABC
from loguru import logger
from ..global_config import CONFIG


class BaseValidator(ABC):
    def __init__(self):
        self.setup_bittensor_objects()
        self.is_running = False
        self.should_exit = False
        self.loop = asyncio.get_event_loop()

    def setup_bittensor_objects(self):
        logger.info("Setting up Bittensor objects.")
        self.wallet = bt.wallet(name=CONFIG.wallet_name, hotkey=CONFIG.wallet_hotkey)
        logger.info(f"Wallet: {self.wallet}")
        self.dendrite = bt.dendrite(wallet=self.wallet)
        logger.info(f"Dendrite: {self.dendrite}")

    @abstractmethod
    async def start_epoch(self):
        pass

    async def run(self):
        logger.info("Starting validator loop.")
        while not self.should_exit:
            try:
                await self.start_epoch()
            except Exception as e:
                logger.error(f"Forward error: {e}")
                traceback.print_exc()
            except KeyboardInterrupt:
                logger.success("Validator killed by keyboard interrupt.")
                exit()

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            logger.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            logger.debug("Started")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            logger.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            logger.debug("Stopped")
