import argparse
import asyncio
import logging
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Tuple

import base  # noqa
import bittensor as bt
import cortext
import torch
import wandb
from base_validator import BaseValidator
from cortext import utils
from embeddings_validator import EmbeddingsValidator
from envparse import env
from image_validator import ImageValidator
from text_validator import TextValidator
from weight_setter import WeightSetter

text_vali = None
image_vali = None
embed_vali = None
metagraph = None
wandb_runs = {}


def get_config() -> bt.config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--netuid", type=int, default=18)
    parser.add_argument("--wandb_off", action="store_false", dest="wandb_on")
    parser.add_argument("--axon.port", type=int, default=8000)
    parser.set_defaults(wandb_on=True)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)
    config = bt.config(parser)
    _args = parser.parse_args()
    full_path = Path(
        f"{config.logging.logging_dir}/{config.wallet.name}/{config.wallet.hotkey}/netuid{config.netuid}/validator"
    ).expanduser()
    config.full_path = str(full_path)
    full_path.mkdir(parents=True, exist_ok=True)
    return config


def init_wandb(config, my_uid, wallet: bt.wallet):
    if not config.wandb_on:
        return

    run_name = f"validator-{my_uid}-{cortext.__version__}"
    config.uid = my_uid
    config.hotkey = wallet.hotkey.ss58_address
    config.run_name = run_name
    config.version = cortext.__version__
    config.type = "validator"

    # Initialize the wandb run for the single project
    run = wandb.init(
        name=run_name, project=cortext.PROJECT_NAME, entity="cortex-t", config=config, dir=config.full_path, reinit=True
    )

    # Sign the run to ensure it's from the correct hotkey
    signature = wallet.hotkey.sign(run.id.encode()).hex()
    config.signature = signature
    wandb.config.update(config, allow_val_change=True)

    bt.logging.success(f"Started wandb run for project '{cortext.PROJECT_NAME}'")


def initialize_components(config: bt.config):
    global metagraph
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running validator for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint}")
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)
    dendrite = bt.dendrite(wallet=wallet)
    my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(
            f"Your validator: {wallet} is not registered to chain connection: "
            f"{subtensor}. Run btcli register --netuid 18 and try again."
        )
        sys.exit()
    return wallet, subtensor, dendrite, my_uid


def initialize_validators(vali_config, test=False):
    global text_vali, image_vali, embed_vali

    text_vali = TextValidator(**vali_config)
    image_vali = ImageValidator(**vali_config)
    embed_vali = EmbeddingsValidator(**vali_config)
    bt.logging.info("initialized_validators")


def main(test=False) -> None:
    config = get_config()
    wallet, subtensor, dendrite, my_uid = initialize_components(config)
    validator_config = {"dendrite": dendrite, "config": config, "subtensor": subtensor, "wallet": wallet}
    initialize_validators(validator_config, test)
    init_wandb(config, my_uid, wallet)
    loop = asyncio.get_event_loop()
    weight_setter = WeightSetter(loop, dendrite, subtensor, config, wallet, text_vali, image_vali, embed_vali)
    state_path = os.path.join(config.full_path, "state.json")
    utils.get_state(state_path)
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        bt.logging.info("Keyboard interrupt detected. Exiting validator.")
    finally:
        state = utils.get_state(state_path)
        utils.save_state_to_file(state, state_path)
        if config.wandb_on:
            wandb.finish()


if __name__ == "__main__":
    main()
