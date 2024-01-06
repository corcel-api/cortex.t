import argparse
import asyncio
import traceback
from pathlib import Path

import bittensor as bt
import wandb
from aiohttp import web
from aiohttp.web_response import Response
from cortex_t.validators.image_validator import ImageValidator
from cortex_t.validators.embeddings_validator import EmbeddingsValidator
from cortex_t.validators.text_validator import TextValidator, TestTextValidator
from envparse import env

from cortex_t import template
from cortex_t.template import utils
import sys

from cortex_t.validators.weight_setter import WeightSetter, TestWeightSetter

text_vali = None
image_vali = None
embed_vali = None
metagraph = None
wandb_runs = {}
# organic requests are scored, the tasks are stored in this queue
# for later being consumed by `query_synapse` cycle:
organic_scoring_tasks = set()
EXPECTED_ACCESS_KEY = env('EXPECTED_ACCESS_KEY', default='hello')


def get_config() -> bt.config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--netuid", type=int, default=18)
    parser.add_argument('--wandb_off', action='store_false', dest='wandb_on')
    parser.add_argument('--http_port', type=int, default=8000)
    parser.set_defaults(wandb_on=True)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
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

    run_name = f'validator-{my_uid}-{template.__version__}'
    config.uid = my_uid
    config.hotkey = wallet.hotkey.ss58_address
    config.run_name = run_name
    config.version = template.__version__
    config.type = 'validator'

    # Initialize the wandb run for the single project
    run = wandb.init(
        name=run_name,
        project=template.PROJECT_NAME,
        entity='cortex-t',
        config=config,
        dir=config.full_path,
        reinit=True
    )

    # Sign the run to ensure it's from the correct hotkey
    signature = wallet.hotkey.sign(run.id.encode()).hex()
    config.signature = signature
    wandb.config.update(config, allow_val_change=True)

    bt.logging.success(f"Started wandb run for project '{template.PROJECT_NAME}'")


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

    text_vali = (TextValidator if not test else TestTextValidator)(**vali_config)
    image_vali = ImageValidator(**vali_config)
    embed_vali = EmbeddingsValidator(**vali_config)
    bt.logging.info("initialized_validators")


async def process_text_validator(request: web.Request):
    # Check access key
    access_key = request.headers.get("access-key")
    if access_key != EXPECTED_ACCESS_KEY:
        return Response(status=401, reason="Invalid access key")

    try:
        messages_dict = {int(k): [{'role': 'user', 'content': v}] for k, v in (await request.json()).items()}
    except ValueError:
        return Response(status=400)

    response = web.StreamResponse()
    await response.prepare(request)

    uid_to_response = dict.fromkeys(messages_dict, "")
    try:
        async for uid, content in text_vali.organic(validator_app.weight_setter.metagraph, messages_dict):
            uid_to_response[uid] += content
            await response.write(content.encode())
        validator_app.weight_setter.register_text_validator_organic_query(
            uid_to_response, {k: v[0]['content'] for k, v in messages_dict.items()}
        )
    except Exception:
        bt.logging.error(f'Encountered in {process_text_validator.__name__}:\n{traceback.format_exc()}')
        await response.write(b'<<internal error>>')

    return response


class ValidatorApplication(web.Application):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.weight_setter: WeightSetter | None = None


validator_app = ValidatorApplication()
validator_app.add_routes([web.post('/text-validator/', process_text_validator)])


def main(run_aio_app=True, test=False) -> None:
    config = get_config()
    wallet, subtensor, dendrite, my_uid = initialize_components(config)
    validator_config = {
        "dendrite": dendrite,
        "config": config,
        "subtensor": subtensor,
        "wallet": wallet
    }
    initialize_validators(validator_config, test)
    init_wandb(config, my_uid, wallet)
    loop = asyncio.get_event_loop()

    weight_setter = (WeightSetter if not test else TestWeightSetter)(
        loop, dendrite, subtensor, config, wallet, text_vali, image_vali, embed_vali)
    validator_app.weight_setter = weight_setter

    if run_aio_app:
        try:
            web.run_app(validator_app, port=config.http_port, loop=loop)
        except KeyboardInterrupt:
            bt.logging.info("Keyboard interrupt detected. Exiting validator.")
        finally:
            state = utils.get_state()
            utils.save_state_to_file(state)
            if config.wandb_on:
                wandb.finish()


if __name__ == "__main__":
    main()
