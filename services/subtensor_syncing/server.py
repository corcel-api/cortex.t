from dotenv import load_dotenv

load_dotenv()

import bittensor as bt
import os
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import time
from cortext import CONFIG
import httpx
import traceback
from fastapi import FastAPI, APIRouter
import uvicorn
import numpy as np


class AutoSyncSubtensor:
    def __init__(self):
        self.subtensor = bt.Subtensor(network=CONFIG.subtensor_network)
        self.metagraph = self.subtensor.metagraph(CONFIG.subtensor_netuid)
        self.netuid = CONFIG.subtensor_netuid
        self.wallet = bt.wallet(name=CONFIG.wallet_name, hotkey=CONFIG.wallet_hotkey)
        self.miner_manager_client = httpx.AsyncClient(
            base_url=f"http://{CONFIG.miner_manager.host}:{CONFIG.miner_manager.port}",
        )
        self.uid = 0
        self.sync_executor = ThreadPoolExecutor(max_workers=1)
        self.set_weights_executor = ThreadPoolExecutor(max_workers=1)
        self.sync_executor.submit(self.sync_subtensor)

        self.router = APIRouter()
        self.router.add_api_route(
            "/api/set_weights", self.do_set_weights, methods=["POST"]
        )
        self.router.add_api_route("/api/axons", self.get_axons, methods=["POST"])

        self.app = FastAPI()
        self.app.include_router(self.router)

    def sync_subtensor(self):
        while True:
            logger.info("Syncing subtensor")
            self.metagraph.sync()
            time.sleep(600)

    def get_axons(self, uids: list[int]) -> list[str]:
        axons: list[bt.AxonInfo] = [self.metagraph.axons[uid] for uid in uids]
        return [axon.to_string() for axon in axons]

    async def do_set_weights(self):
        logger.info("Setting weights")
        current_block = self.subtensor.get_current_block()
        last_update = self.metagraph.last_update[self.uid]
        logger.info(f"Current block: {current_block}")
        logger.info(f"Last update: {last_update}")
        logger.info("Getting weights from miner manager")
        response = await self.miner_manager_client.get("/api/weights")
        response_json = response.json()
        weights = response_json["weights"]
        uids = response_json["uids"]
        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=np.array(uids),
            weights=np.array(weights),
            netuid=self.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        (
            uint_uids,
            uint_weights,
        ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        logger.info(f"Setting weights for {self.uid}")
        logger.info(f"Current block: {current_block}")
        if current_block > last_update + CONFIG.subtensor_tempo:
            try:
                future = self.set_weights_executor.submit(
                    self.subtensor.set_weights,
                    netuid=self.netuid,
                    wallet=self.wallet,
                    uids=uint_uids,
                    weights=uint_weights,
                )
                success, msg = future.result(timeout=120)
                if not success:
                    logger.error(f"Failed to set weights: {msg}")
                    return {"success": False, "message": msg}
                else:
                    logger.info(f"Set weights result: {success}")
                    return {"success": True, "message": msg}
            except Exception as e:
                logger.error(f"Failed to set weights: {e}")
                traceback.print_exc()
        else:
            logger.info(
                f"Not setting weights because current block {current_block} is not greater than last update {last_update} + tempo {CONFIG.subtensor_tempo}"
            )


if __name__ == "__main__":
    auto_sync_subtensor = AutoSyncSubtensor()
    uvicorn.run(
        auto_sync_subtensor.app,
        host=CONFIG.w_subtensor.host,
        port=CONFIG.w_subtensor.port,
    )
